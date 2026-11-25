"""Fixed-storage XPU graph replay, independent reference cache, exact RoPE and fused projections."""
# SPDX-License-Identifier: GPL-3.0-only
# Portions of the attention/MLP integration adapt ComfyUI's implementation.
# Copyright in upstream portions remains with their respective ComfyUI contributors.
# Upstream: https://github.com/Comfy-Org/ComfyUI/blob/8f04fd6ecf4a548937971e840af2a985529fab76/comfy/text_encoders/llama.py
# Modified 2026-11-25 UTC for XPU graph replay, fused projections and fixed-storage KV decoding.
# See LICENSE and NOTICE.md.
import logging
import time
import torch
from comfy.ops import QuantizedTensor, run_every_op
from .settings import TAG, MISSING

def run_graph_decode(owner, args, kwargs, device, check_all, *, original_generate, eligible_modules, fusion_ops, lut_ops, multimodal=False, set_projection_enabled=None, resident_check=None):
    from .kernels.rope import _qwen_mixed_rope
    from comfy.text_encoders.llama import apply_rope, rope_matrix
    exact_rope_counts = {'custom': 0, 'fallback': 0}

    def decode_rope(xq, xk, freqs_cis):
        rotation = rope_matrix(freqs_cis)
        eligible = xq.device.type == 'xpu' and xq.device == xk.device and (xq.device == rotation.device) and (xq.dtype == torch.bfloat16) and (xk.dtype == torch.bfloat16) and (rotation.dtype == torch.float32) and (xq.shape == (1, 32, 1, 128)) and (xk.shape == (1, 8, 1, 128)) and (rotation.numel() == 256) and xq.is_contiguous() and xk.is_contiguous() and rotation.is_contiguous()
        if eligible:
            exact_rope_counts['custom'] += 1
            return _qwen_mixed_rope(xq, xk, rotation, form=3)
        exact_rope_counts['fallback'] += 1
        return apply_rope(xq, xk, freqs_cis)
    import torch.nn.functional as F
    model = owner.model
    original_forward = model.forward
    original_logits = owner.logits
    original_sample_token = owner.sample_token
    capture_argmax = True
    import inspect
    bound = inspect.signature(original_generate).bind(owner, *args, **kwargs)
    bound.apply_defaults()
    greedy = not bound.arguments['do_sample'] or bound.arguments['temperature'] == 0.0
    if not greedy:
        capture_argmax = False
        logging.info('%s: ARGMAX using standard sampling', TAG)
    argmax_calls = 0
    graph_token = None
    replacements = []
    graph = None
    active = False
    graph_hidden = None
    graph_logits = None
    position = None
    key_positions = None
    shared_attention_mask = None
    static_source = None
    static_position_ids = None
    replay_count = 0
    steady_start = None

    def replace_method(obj, name, method):
        previous = obj.__dict__.get(name, MISSING)
        replacements.append((obj, name, previous))
        setattr(obj, name, method)
    fusion_counts = {'qkv': 0, 'mlp': 0}
    reference_cache = None
    native_reference_cache = None
    mm_prefilled = False
    mm_disabled = False
    mm_cache_signature = None
    mm_expected_index = None
    bits = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
    silu_lut = F.silu(bits.view(torch.bfloat16)).contiguous()
    logging.info('%s: Exact SiLU lookup table enabled, bytes=%d', TAG, silu_lut.numel() * silu_lut.element_size())

    def projection_data(module, x, shape):
        if id(module) not in eligible_modules:
            return None
        w = module.weight
        if not isinstance(w, QuantizedTensor):
            return None
        if x.dtype != torch.bfloat16 or x.device.type != 'xpu' or x.numel() != shape[1] or (not x.is_contiguous()) or (tuple(w._qdata.shape) != shape) or (w._qdata.dtype != torch.float8_e4m3fn) or (w._qdata.device != x.device) or (not w._qdata.is_contiguous()) or (w.params.scale.numel() != 1) or (w.params.scale.device != x.device) or (getattr(module, '_v', None) is not None) or (getattr(module, 'weight_lowvram_function', None) is not None) or (getattr(module, 'bias_lowvram_function', None) is not None) or getattr(module, 'comfy_force_forward', False) or (getattr(module, 'bias', None) is not None) or (getattr(module, 'input_scale', None) is not None) or (getattr(module, 'pre_quant_scale', None) is not None) or len(getattr(module, 'weight_function', [])) or len(getattr(module, 'bias_function', [])):
            return None
        return (w._qdata.view(torch.uint8), w.params.scale.float())

    def allocate_projection(x, n):
        return torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)

    def try_fused_qkv(attn, x):
        q = projection_data(attn.q_proj, x, (4096, 2560))
        k = projection_data(attn.k_proj, x, (1024, 2560))
        v = projection_data(attn.v_proj, x, (1024, 2560))
        if any((z is None for z in (q, k, v))):
            return None
        for _ in range(3):
            run_every_op()
        qo = allocate_projection(x, 4096)
        ko = allocate_projection(x, 1024)
        vo = allocate_projection(x, 1024)
        fusion_ops.qkv(x, q[0], k[0], v[0], q[1], k[1], v[1], qo, ko, vo, 128)
        fusion_counts['qkv'] += 1
        return (qo, ko, vo)

    def make_mlp(mlp, original):

        def forward(x, norm=None):
            if not active or mlp.merged_mlp or mlp.activation is not F.silu:
                return original(x, norm=norm)
            normalized = norm(x) if norm is not None else x
            g = projection_data(mlp.gate_proj, normalized, (9728, 2560))
            u = projection_data(mlp.up_proj, normalized, (9728, 2560))
            if g is None or u is None:
                return original(normalized)
            for _ in range(2):
                run_every_op()
            y = allocate_projection(normalized, 9728)
            lut_ops.mlp(normalized, g[0], u[0], g[1], u[1], silu_lut, y, 128)
            fusion_counts['mlp'] += 1
            return mlp.down_proj(y)
        return forward

    def make_attention(attn, original):

        def forward(hidden_states, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None, sliding_window=None):
            if not active:
                return original(hidden_states, attention_mask=attention_mask, freqs_cis=freqs_cis, optimized_attention=optimized_attention, past_key_value=past_key_value, sliding_window=sliding_window)
            assert hidden_states.shape[:2] == (1, 1)
            assert isinstance(past_key_value, tuple)
            assert sliding_window is None
            assert attention_mask is None
            if attn.merged_qkv:
                q, k, v = attn.qkv_proj(hidden_states).split((attn.inner_size, attn.kv_size, attn.kv_size), dim=-1)
            else:
                projected = try_fused_qkv(attn, hidden_states)
                if projected is not None:
                    q, k, v = projected
                else:
                    q = attn.q_proj(hidden_states)
                    k = attn.k_proj(hidden_states)
                    v = attn.v_proj(hidden_states)
            q = q.view(1, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(1, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = v.view(1, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            if attn.q_norm is not None:
                q = attn.q_norm(q)
            if attn.k_norm is not None:
                k = attn.k_norm(k)
            q, k = decode_rope(q, k, freqs_cis=freqs_cis)
            key_cache, value_cache, index = past_key_value
            scatter_index = position.view(1, 1, 1, 1).expand_as(k)
            key_cache.scatter_(2, scatter_index, k)
            value_cache.scatter_(2, scatter_index, v)
            mask = shared_attention_mask
            output = F.scaled_dot_product_attention(q, key_cache, value_cache, attn_mask=mask, dropout_p=0.0, is_causal=False, enable_gqa=attn.num_heads != attn.num_kv_heads)
            output = output.transpose(1, 2).reshape(1, 1, attn.inner_size)
            return (attn.o_proj(output), (key_cache, value_cache, index + 1))
        return forward

    def cached_logits(x):
        if graph_hidden is not None and x is graph_hidden:
            return graph_logits
        return original_logits(x)

    def cached_sample_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history, generator, do_sample=True, presence_penalty=0.0, penalty_mask=None):
        nonlocal argmax_calls
        if capture_argmax and graph_token is not None and (graph_logits is not None) and (not do_sample or temperature == 0.0) and (logits.shape == graph_logits[:, -1].shape) and (logits.data_ptr() == graph_logits[:, -1].data_ptr()):
            argmax_calls += 1
            return graph_token
        return original_sample_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history, generator, do_sample=do_sample, presence_penalty=presence_penalty, penalty_mask=penalty_mask)

    def graph_forward(*fa, **fk):
        nonlocal graph_token
        nonlocal graph, active
        nonlocal graph_hidden, graph_logits
        nonlocal position, key_positions
        nonlocal static_source, static_position_ids
        nonlocal replay_count, steady_start
        nonlocal reference_cache, native_reference_cache
        nonlocal mm_prefilled, mm_disabled, mm_cache_signature, mm_expected_index
        incoming = fk.get('embeds')
        cache = fk.get('past_key_values')
        if multimodal:
            from .multimodal import decode_contract
            if not mm_prefilled:
                # No optimized projection or attention is used here.
                torch.xpu.synchronize(device)
                t0 = time.perf_counter()
                result = original_forward(*fa, **fk)
                torch.xpu.synchronize(device)
                mm_prefilled = True
                logging.info('%s: MM_PREFILL tokens=%d seconds=%.3f cache=%s',
                             TAG, incoming.shape[1], time.perf_counter() - t0,
                             type(result[2][0]).__name__)
                return result
            if mm_disabled:
                return original_forward(*fa, **fk)
            reason, signature = decode_contract(fa, fk, device, len(model.layers))
            if reason is None and graph is None:
                reason = resident_check()
            if reason is None and graph is not None:
                if signature != mm_cache_signature or cache[0][2] != mm_expected_index:
                    reason = 'cache storage or index changed'
            if reason is not None:
                if graph is not None:
                    # Do not replay a graph against storage different from its capture.
                    raise RuntimeError('Multimodal graph contract changed: ' + reason)
                mm_disabled = True
                logging.info('%s: MM fallback %s', TAG, reason)
                return original_forward(*fa, **fk)
            if graph is None:
                mm_cache_signature = signature
                logging.info('%s: MM_DECODE cache_index=%d rope_position=%d capacity=%d',
                             TAG, cache[0][2], fk['position_ids'].item(), cache[0][0].shape[2])
                set_projection_enabled(True)
        if incoming is None or incoming.shape[:2] != (1, 1) or (not cache) or (not isinstance(cache[0], tuple)) or (cache[0][2] == 0):
            return original_forward(*fa, **fk)
        index = cache[0][2]
        capacity = cache[0][0].shape[2]
        first = graph is None
        assert index < capacity
        incoming_position_ids = fk.get('position_ids')
        if first:
            position = torch.empty((), device=device, dtype=torch.long)
            key_positions = torch.arange(capacity, device=device).view(1, 1, 1, capacity)
            static_source = torch.empty_like(incoming)
            work = torch.empty_like(incoming)
            if incoming_position_ids is None:
                static_position_ids = torch.empty((1, 1), device=device, dtype=torch.long)
            else:
                static_position_ids = torch.empty_like(incoming_position_ids)
            for key, value, _ in cache:
                key[:, :, index:].zero_()
                value[:, :, index:].zero_()
        position.fill_(index)
        static_source.copy_(incoming)
        if incoming_position_ids is None:
            static_position_ids.fill_(index)
        else:
            static_position_ids.copy_(incoming_position_ids)
        reference_logits = None
        native_logits = None
        if multimodal and check_all:
            # Independent native eager cache, seeded by the same unmodified prefill.
            if native_reference_cache is None:
                native_reference_cache = [(k.clone(), v.clone(), i) for k, v, i in cache]
            native_kwargs = dict(fk)
            native_kwargs['embeds'] = incoming.clone()
            native_kwargs['past_key_values'] = native_reference_cache
            set_projection_enabled(False)
            try:
                native_result = original_forward(*fa, **native_kwargs)
                native_logits = original_logits(native_result[0]).clone()
                native_reference_cache = native_result[2]
                del native_result
            finally:
                set_projection_enabled(True)
        if first or check_all:
            reference_kwargs = dict(fk)
            reference_kwargs['embeds'] = incoming.clone()
            if check_all:
                if reference_cache is None:
                    reference_cache = [(k.clone(), v.clone(), i) for k, v, i in cache]
                reference_kwargs['past_key_values'] = reference_cache
            reference = original_forward(*fa, **reference_kwargs)
            reference_logits = original_logits(reference[0]).clone()
            if check_all:
                reference_cache = reference[2]
            del reference
        if first:
            static_kwargs = dict(fk)
            static_kwargs['embeds'] = work
            static_kwargs['position_ids'] = static_position_ids

            def workload():
                nonlocal active
                nonlocal shared_attention_mask
                active = True
                try:
                    work.copy_(static_source)
                    work_kwargs = dict(static_kwargs)
                    freqs = work_kwargs.get('freqs_cis')
                    if freqs is None:
                        freqs = model.compute_freqs_cis(static_position_ids, device)
                    work_kwargs['freqs_cis'] = rope_matrix(freqs)
                    shared_attention_mask = key_positions <= position.view(1, 1, 1, 1)
                    hidden, _, _ = original_forward(*fa, **work_kwargs)
                    logits = original_logits(hidden)
                    token = logits[:, -1].argmax(dim=-1, keepdim=True) if capture_argmax else None
                    return (hidden, logits, token)
                finally:
                    active = False
            torch.xpu.synchronize(device)
            stream = torch.xpu.Stream()
            with torch.xpu.stream(stream):
                for _ in range(2):
                    workload()
            torch.xpu.synchronize(device)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=stream):
                graph_hidden, graph_logits, graph_token = workload()
            logging.info('%s: GRAPH_DECODE captured position=%d capacity=%d', TAG, index, capacity)
        graph.replay()
        replay_count += 1
        if reference_logits is not None:
            error = (torch.linalg.vector_norm(graph_logits.float() - reference_logits.float()) / torch.linalg.vector_norm(reference_logits.float()).clamp_min(1e-12)).item()
            same_token = graph_logits.argmax(dim=-1).item() == reference_logits.argmax(dim=-1).item()
            finite = torch.isfinite(graph_logits).all().item()
            logging.info('%s: GRAPH_DECODE check position=%d relative_L2=%.8f same_token=%s finite=%s', TAG, index, error, same_token, finite)
            if check_all:
                mismatches = (graph_logits != reference_logits).sum().item()
                logging.info('%s: LUT_VALIDATION position=%d logit_mismatches=%d', TAG, index, mismatches)
            if not finite or error >= 0.01 or (not same_token):
                raise RuntimeError('Graph decode correctness check failed')
        if native_logits is not None:
            native_error = (torch.linalg.vector_norm(graph_logits.float() - native_logits.float()) /
                            torch.linalg.vector_norm(native_logits.float()).clamp_min(1e-12)).item()
            native_same = graph_logits.argmax(dim=-1).item() == native_logits.argmax(dim=-1).item()
            native_finite = torch.isfinite(native_logits).all().item()
            logging.info('%s: MM_NATIVE_CHECK position=%d relative_L2=%.8f same_token=%s finite=%s',
                         TAG, index, native_error, native_same, native_finite)
            if not native_finite:
                raise RuntimeError('Nonfinite original native multimodal reference')
        if capture_argmax and reference_logits is not None:
            expected = reference_logits[:, -1].argmax(dim=-1, keepdim=True)
            if not torch.equal(graph_token, expected):
                raise RuntimeError('Captured argmax correctness check failed')
        if first:
            torch.xpu.synchronize(device)
            steady_start = time.perf_counter()
        updated_cache = [(key, value, index + 1) for key, value, _ in cache]
        if multimodal:
            mm_expected_index = index + 1
        return (graph_hidden, None, updated_cache)
    try:
        for layer in model.layers:
            attn = layer.self_attn
            replace_method(attn, 'forward', make_attention(attn, attn.forward))
        for layer in model.layers:
            mlp = layer.mlp
            replace_method(mlp, 'forward', make_mlp(mlp, mlp.forward))
        replace_method(model, 'forward', graph_forward)
        replace_method(owner, 'logits', cached_logits)
        if capture_argmax:
            replace_method(owner, 'sample_token', cached_sample_token)
        tokens = original_generate(owner, *args, **kwargs)
        import hashlib
        import json
        digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        logging.info('%s: TOKEN_DIGEST count=%d sha256=%s', TAG, len(tokens), digest)
        if capture_argmax:
            logging.info('%s: ARGMAX captured_calls=%d', TAG, argmax_calls)
        torch.xpu.synchronize(device)
        if steady_start is not None and replay_count > 1:
            elapsed = time.perf_counter() - steady_start
            logging.info('%s: GRAPH_DECODE replays=%d steady_seconds=%.3f steady_tokens_per_second=%.3f validation=%s', TAG, replay_count, elapsed, (replay_count - 1) / max(elapsed, 1e-09), check_all)
        logging.info('%s: EXACT_ROPE custom_calls=%d fallback_calls=%d', TAG, exact_rope_counts['custom'], exact_rope_counts['fallback'])
        logging.info('%s: FUSION qkv_calls=%d mlp_calls=%d', TAG, fusion_counts['qkv'], fusion_counts['mlp'])
        return tokens
    finally:
        for obj, name, previous in reversed(replacements):
            if previous is MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)
