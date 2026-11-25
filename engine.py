"""Native Qwen3-VL-4B XPU acceleration; multimodal candidate is opt-in."""
import functools
import json
import math
import inspect
import logging
import time

import torch
from comfy.ops import QuantizedTensor, MixedPrecisionOp, run_every_op
from comfy.text_encoders.llama import BaseGenerate as GenerationClass
from .decoder import run_graph_decode
from .multimodal import trace_native, resident_reason
from .native import load_native_ops
from .settings import TAG, MODE_FILE, CALIBRATION_FILE, MISSING

original_generate = GenerationClass.generate
generate_signature = inspect.signature(original_generate)

@functools.wraps(original_generate)
def generate(self, *args, **kwargs):
    try:
        mode = MODE_FILE.read_text().strip().lower()
    except FileNotFoundError:
        mode = 'off'
    if mode not in ('on', 'validate', 'mm_trace', 'mm_validate', 'mm_on'):
        return original_generate(self, *args, **kwargs)
    config = getattr(self.model, 'config', None)
    if type(config).__name__ != 'Qwen3VL_4BConfig':
        return original_generate(self, *args, **kwargs)
    arguments = generate_signature.bind(self, *args, **kwargs).arguments
    embeds = arguments.get('embeds')
    if embeds is None or embeds.device.type != 'xpu' or (embeds.ndim not in (2, 3)) or (embeds.ndim == 3 and embeds.shape[0] != 1) or (arguments.get('execution_dtype') not in (None, torch.bfloat16)):
        return original_generate(self, *args, **kwargs)
    multimodal = any(arguments.get(k) is not None for k in ('deepstack_embeds', 'position_ids', 'visual_pos_masks'))
    if mode in ('on', 'validate'):
        # Preserve the installed text-only routing.
        if arguments.get('deepstack_embeds') is not None:
            return original_generate(self, *args, **kwargs)
        multimodal = False
    elif multimodal:
        if mode == 'mm_trace':
            return trace_native(self, args, kwargs, original_generate, arguments)
        # Only the observed Qwen3-VL input contract is supported initially.
        pos = arguments.get('position_ids')
        deep = arguments.get('deepstack_embeds')
        mask = arguments.get('visual_pos_masks')
        if (embeds.ndim != 3 or embeds.shape[-1] != 2560 or embeds.shape[1] < 2
                or pos is None or tuple(pos.shape) != (3, embeds.shape[1])
                or pos.device != embeds.device or not isinstance(deep, (list, tuple))
                or len(deep) != 3 or mask is None
                or tuple(mask.shape) != (1, embeds.shape[1])
                or mask.dtype != torch.bool or mask.device != embeds.device
                or arguments.get('initial_input_ids') is not None):
            logging.info('%s: MM fallback unsupported prompt contract', TAG)
            return original_generate(self, *args, **kwargs)
    try:
        import triton
        from .kernels.fp8_gemv import fused_gemv
        from .kernels.fp8_gemv_tiled import tiled_gemv
        sycl_gemv, fusion_ops, lut_ops = load_native_ops()
        calibration = json.loads(CALIBRATION_FILE.read_text())
        if not isinstance(calibration, dict) or not calibration:
            raise ValueError('Missing calibration data')
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        logging.warning('%s: initialization unavailable, using native generation: %s', TAG, exc)
        return original_generate(self, *args, **kwargs)
    device = embeds.device
    replacements = []
    stats = {'calls': 0, 'hybrid_calls': 0, 'hybrid_shapes': {}, 'rejected': {}}

    projection_enabled = not multimodal

    def set_projection_enabled(value):
        nonlocal projection_enabled
        projection_enabled = value

    def make_forward(name, module, original_forward):
        static_scale = None
        inverse_scale = None
        if name in calibration:
            peak = float(calibration[name])
            value = 2.0 ** math.ceil(math.log2(max(peak * 2.0 / 448.0, 2.0 ** (-16))))
            static_scale = torch.tensor(value, device=device, dtype=torch.float32)
            inverse_scale = 1.0 / value

        def fast_forward(input, *a, **kw):
            if not projection_enabled:
                return original_forward(input, *a, **kw)
            weight = module.weight
            reason = None
            if input.dtype not in (torch.float16, torch.bfloat16):
                reason = 'dtype'
            elif input.numel() != input.shape[-1]:
                reason = 'prefill_or_batch'
            elif not isinstance(weight, QuantizedTensor):
                reason = 'weight_type'
            elif weight._qdata.device != input.device:
                reason = 'weight_device'
            elif weight._qdata.ndim != 2:
                reason = 'weight_dimensions'
            elif weight._qdata.dtype != torch.float8_e4m3fn:
                reason = 'weight_dtype'
            elif weight.params.scale.numel() != 1:
                reason = 'weight_scale'
            elif weight.params.scale.device != input.device:
                reason = 'scale_device'
            elif input.shape[-1] != weight._qdata.shape[1]:
                reason = 'shape'
            elif getattr(module, '_v', None) is not None:
                reason = 'vbar'
            elif getattr(module, 'weight_lowvram_function', None) is not None:
                reason = 'lowvram'
            elif getattr(module, 'comfy_force_forward', False):
                reason = 'force_forward'
            elif getattr(module, 'bias_lowvram_function', None) is not None:
                reason = 'bias_lowvram'
            elif getattr(module, 'bias', None) is not None:
                reason = 'bias'
            elif getattr(module, 'input_scale', None) is not None:
                reason = 'input_scale'
            elif getattr(module, 'pre_quant_scale', None) is not None:
                reason = 'pre_quant_scale'
            elif len(getattr(module, 'weight_function', [])):
                reason = 'weight_function'
            elif len(getattr(module, 'bias_function', [])):
                reason = 'bias_function'
            elif a or kw:
                reason = 'extra_arguments'
            if reason is not None:
                rejected = stats['rejected']
                rejected[reason] = rejected.get(reason, 0) + 1
                return original_forward(input, *a, **kw)
            run_every_op()
            x = input.reshape(1, input.shape[-1])
            if x.dtype == torch.bfloat16 and x.is_contiguous() and weight._qdata.is_contiguous():
                n, k = weight._qdata.shape
                shape = (n, k)
                full_width_shapes = ((4096, 2560), (1024, 2560), (2560, 4096))
                if shape in full_width_shapes or shape == (2560, 9728) or shape == (9728, 2560):
                    y = torch.empty((1, n), device=input.device, dtype=input.dtype)
                    raw = weight._qdata.view(torch.uint8)
                    ws = weight.params.scale.float()
                    if shape == (9728, 2560) or shape in ((4096, 2560), (1024, 2560)):
                        wg = 128 if shape == (9728, 2560) else 256
                        sycl_gemv(x, raw, ws, y, wg)
                        stats['sycl_calls'] = stats.get('sycl_calls', 0) + 1
                    elif shape in full_width_shapes:
                        fused_gemv[n,](x, raw, ws, y, N=n, K=k, BM=1, BK=triton.next_power_of_2(k), num_warps=8)
                    else:
                        tiled_gemv[n,](x, raw, ws, y, N=n, K=k, BM=1, BK=256, num_warps=4)
                    stats['calls'] += 1
                    stats['hybrid_calls'] += 1
                    shapes = stats['hybrid_shapes']
                    shapes[str(shape)] = shapes.get(str(shape), 0) + 1
                    return y.reshape(*input.shape[:-1], n)
            if static_scale is None:
                rejected = stats['rejected']
                rejected['uncalibrated'] = rejected.get('uncalibrated', 0) + 1
                return original_forward(input, *a, **kw)
            activation_scale = static_scale
            activation = (x * inverse_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            result = torch._scaled_mm(activation, weight._qdata.t(), scale_a=activation_scale, scale_b=weight.params.scale.float().reshape(()), out_dtype=input.dtype)
            stats['calls'] += 1
            return result.reshape(*input.shape[:-1], weight._qdata.shape[0])
        return fast_forward
    try:
        for name, module in self.model.layers.named_modules():
            if not isinstance(module, MixedPrecisionOp):
                continue
            if getattr(module, 'quant_format', None) != 'float8_e4m3fn':
                continue
            if getattr(module, '_full_precision_mm_config', False):
                continue
            if getattr(module, 'bias', None) is not None:
                continue
            if getattr(module, 'input_scale', None) is not None:
                continue
            if getattr(module, 'pre_quant_scale', None) is not None:
                continue
            previous = module.__dict__.get('forward', MISSING)
            original_forward = module.forward
            module.forward = make_forward(name, module, original_forward)
            replacements.append((module, previous))
        logging.info('%s: mode=%s eligible_layers=%d embeds_dtype=%s', TAG, mode, len(replacements), embeds.dtype)
        torch.xpu.synchronize(device)
        start = time.perf_counter()
        eligible_modules = {id(module) for module, _ in replacements}
        tokens = run_graph_decode(
            self, args, kwargs, device,
            mode == 'validate' or (multimodal and mode == 'mm_validate'),
            original_generate=original_generate, eligible_modules=eligible_modules,
            fusion_ops=fusion_ops, lut_ops=lut_ops,
            multimodal=multimodal,
            set_projection_enabled=set_projection_enabled,
            resident_check=lambda: resident_reason(self, device, eligible_modules))
        torch.xpu.synchronize(device)
        elapsed = time.perf_counter() - start
        logging.info('%s: mode=%s tokens=%d seconds=%.3f tokens_per_second=%.3f fast_calls=%d', TAG, mode, len(tokens), elapsed, len(tokens) / max(elapsed, 1e-09), stats['calls'])
        logging.info('%s: SYCL custom_calls=%d', TAG, stats.get('sycl_calls', 0))
        logging.info('%s: HYBRID custom_calls=%d shapes=%s', TAG, stats['hybrid_calls'], stats['hybrid_shapes'])
        logging.info('%s: rejected_calls=%s', TAG, stats['rejected'])
        return tokens
    finally:
        for module, previous in reversed(replacements):
            if previous is MISSING:
                delattr(module, 'forward')
            else:
                module.forward = previous
