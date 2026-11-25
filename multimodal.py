"""Opt-in multimodal contract checks and native trace for the XPU decoder."""
import logging
import time
import torch
from comfy.ops import QuantizedTensor
from .settings import TAG, MISSING


def resident_reason(owner, device, eligible_modules):
    model = owner.model
    # Initial experiment supports entirely resident, unmodified language weights.
    for name, module in model.named_modules():
        if (getattr(module, '_v', None) is not None
                or getattr(module, 'weight_lowvram_function', None) is not None
                or getattr(module, 'bias_lowvram_function', None) is not None
                or getattr(module, 'comfy_force_forward', False)
                or getattr(module, 'weight_function', None)
                or getattr(module, 'bias_function', None)):
            return 'dynamic or patched module: ' + name
    for name, param in model.named_parameters():
        if param.device != device:
            return 'nonresident parameter: ' + name
    head = model.embed_tokens.weight
    if isinstance(head, QuantizedTensor) or head.dtype != torch.bfloat16:
        return 'tied embedding is not resident BF16'
    for i, layer in enumerate(model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        if attn.merged_qkv or mlp.merged_mlp:
            return 'merged projection'
        projections = (
            (attn.q_proj, (4096, 2560)), (attn.k_proj, (1024, 2560)),
            (attn.v_proj, (1024, 2560)), (attn.o_proj, (2560, 4096)),
            (mlp.gate_proj, (9728, 2560)), (mlp.up_proj, (9728, 2560)),
            (mlp.down_proj, (2560, 9728)))
        for module, shape in projections:
            w = module.weight
            if (id(module) not in eligible_modules
                    or not isinstance(w, QuantizedTensor)
                    or w._qdata.dtype != torch.float8_e4m3fn
                    or tuple(w._qdata.shape) != shape
                    or w._qdata.device != device or not w._qdata.is_contiguous()
                    or w.params.scale.numel() != 1 or w.params.scale.device != device):
                return 'ineligible projection in layer %d: %s' % (i, shape)
    return None


def decode_contract(fa, fk, device, layers):
    # The installed native generator passes exactly these arguments in XPU decode.
    if (len(fa) != 1 or fa[0] is not None
            or set(fk) - {'embeds', 'attention_mask', 'past_key_values', 'input_ids',
                         'position_ids', 'embeds_info'}):
        return 'unexpected decode arguments', None
    if any(fk.get(k) is not None for k in ('attention_mask', 'input_ids', 'embeds_info')):
        return 'unexpected decode metadata', None
    x, pos, cache = (fk.get(k) for k in ('embeds', 'position_ids', 'past_key_values'))
    if (not torch.is_tensor(x) or tuple(x.shape) != (1, 1, 2560)
            or x.dtype != torch.bfloat16 or x.device != device):
        return 'unsupported decode embeddings', None
    if (not torch.is_tensor(pos) or tuple(pos.shape) != (1, 1)
            or pos.dtype != torch.long or pos.device != device):
        return 'unsupported decode position IDs', None
    if not isinstance(cache, (list, tuple)) or len(cache) != layers:
        return 'unsupported cache list', None
    signature = []
    index, capacity = None, None
    for kv in cache:
        if not isinstance(kv, tuple) or len(kv) != 3:
            return 'unsupported non-tuple cache', None
        k, v, i = kv
        if (not torch.is_tensor(k) or not torch.is_tensor(v)
                or k.ndim != 4 or k.shape[:2] != (1, 8) or k.shape[3] != 128
                or k.shape != v.shape or k.dtype != torch.bfloat16 or v.dtype != k.dtype
                or k.device != device or v.device != device
                or not k.is_contiguous() or not v.is_contiguous()
                or not isinstance(i, int) or not 0 < i < k.shape[2]):
            return 'unsupported cache tensor', None
        if index is None:
            index, capacity = i, k.shape[2]
        elif i != index or k.shape[2] != capacity:
            return 'inconsistent layer cache', None
        signature.append((k.data_ptr(), v.data_ptr(), tuple(k.shape)))
    return None, tuple(signature)


def trace_native(owner, args, kwargs, original_generate, arguments):
    model = owner.model
    device = arguments['embeds'].device
    previous = model.__dict__.get('forward', MISSING)
    original = model.forward
    calls = 0
    prefill_seconds = None

    def traced(*fa, **fk):
        nonlocal calls, prefill_seconds
        step = calls
        calls += 1
        if step > 2:
            return original(*fa, **fk)
        x, pos = fk.get('embeds'), fk.get('position_ids')
        cache = fk.get('past_key_values')
        first = cache[0] if cache else None
        index = first[2] if isinstance(first, tuple) else getattr(first, 'index', None)
        key = first[0] if isinstance(first, tuple) else getattr(first, 'key', None)
        deep = fk.get('deepstack_embeds')
        mask = fk.get('visual_pos_masks')
        logging.info('%s: MM_TRACE step=%d embeds=%s dtype=%s position_shape=%s '
                     'last_position=%s cache=%s index=%s key_shape=%s '
                     'deepstack=%s visual_mask=%s kwargs=%s',
                     TAG, step, tuple(x.shape), x.dtype,
                     tuple(pos.shape) if pos is not None else None,
                     pos[:, -1].tolist() if pos is not None else None,
                     type(first).__name__, index, tuple(key.shape) if key is not None else None,
                     [tuple(d.shape) for d in deep] if deep is not None else None,
                     tuple(mask.shape) if mask is not None else None, sorted(fk))
        if step == 1:
            # Approximate pre-capture suitability; use the same shape/weight checks as the candidate.
            from comfy.ops import MixedPrecisionOp
            eligible = {id(m) for m in model.layers.modules()
                        if isinstance(m, MixedPrecisionOp)
                        and getattr(m, 'quant_format', None) == 'float8_e4m3fn'
                        and not getattr(m, '_full_precision_mm_config', False)
                        and getattr(m, 'bias', None) is None
                        and getattr(m, 'input_scale', None) is None
                        and getattr(m, 'pre_quant_scale', None) is None}
            reason, _ = decode_contract(fa, fk, device, len(model.layers))
            logging.info('%s: MM_TRACE eligible=%d contract=%s residency=%s',
                         TAG, len(eligible), reason or 'ok',
                         resident_reason(owner, device, eligible) or 'ok')
        torch.xpu.synchronize(device)
        start = time.perf_counter()
        result = original(*fa, **fk)
        torch.xpu.synchronize(device)
        elapsed = time.perf_counter() - start
        if step == 0:
            prefill_seconds = elapsed
        logging.info('%s: MM_TRACE step=%d forward_seconds=%.3f', TAG, step, elapsed)
        return result

    model.forward = traced
    try:
        torch.xpu.synchronize(device)
        start = time.perf_counter()
        tokens = original_generate(owner, *args, **kwargs)
        torch.xpu.synchronize(device)
        elapsed = time.perf_counter() - start
        logging.info('%s: MM_TRACE tokens=%d seconds=%.3f prefill_seconds=%s tokens_per_second=%.3f',
                     TAG, len(tokens), elapsed, prefill_seconds, len(tokens) / max(elapsed, 1e-9))
        return tokens
    finally:
        if previous is MISSING:
            delattr(model, 'forward')
        else:
            model.forward = previous
