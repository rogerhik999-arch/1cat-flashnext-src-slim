# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.loading — 加载期热/冷权重物化(两档存储主路径)

(ds-cloud 两档存储补丁)

★本模块的 `process_weights_after_loading` 是**唯一一处与上游同步的方法副本**。
  上游 `nvfp4_sm70_moe.py` 已恢复 1Cat 原版(不再含我们的改动);我们以「子类 +
  文件内直连」接管。该方法因「热/冷钩子与上游循环体交织」无法用 super()
  组合,故按上游同名方法**逐字复制**并注入 5 个具名钩子(实现见 weights.py)。

  同步基线:上游 nvfp4_sm70_moe.py(复制时快照 2015 行;当前上游 2042 行);
  复制来源:批1 现役入口 nvfp4_sm70_moe.py
           (md5 909fdd5063b9159e01fe6ea209f82bce)的方法体 L792-1318。
  ⚠ 已知差异(2026-10-02 审计 F1/F12):本副本落后上游快照,
    上游新增的 grouped_mtp5 分支(nvfp4_sm70_moe.py:1200/1206/1208/1214/1215)
    未同步 ⇒ 本副本不设置 layer.sm70_nvfp4_grouped_mtp5;
    已加 fail-closed(方法内见 [vnext-F1]):VLLM_SM70_NVFP4_MOE_GROUPED_MTP5
    置位时显式 raise,拒绝"开了却静默失效"。
  上游若更新该方法,须按下方 5 个钩子注入点重新同步。

  5 个钩子注入点(相对上游的改动,全部由 hooks 承担):
    #4 resolve_hotcold_state   ← 循环前:解析热/冷分档
    #5 fill_hotcold_w13        ← 循环体:w13 落显存热 / pinned 冷
    #6 fill_hotcold_w2         ← 循环体:w2  落显存热 / pinned 冷
    #7 finalize_hotcold        ← 循环后:落位 tm 权重 / swap 预热 / 冷 UVA
    #8 build_hotcold_ptrs      ← 循环后:分段 strided 指针 + merge
"""
from __future__ import annotations

import os

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import MoEActivation, RoutedExperts
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    unpack_mxfp4_weight,
)

# 上游模块级常量/助手(单一真相源仍在入口;此处仅引用,不复制)。
# H2(2026-10-10):旧版在**模块导入期**从宿主反向 import ⇒ 与「宿主末尾 import 本包」
#   形成模块级环,仅靠导入顺序成立(见 docs/代码审计-20261010.md §3-H2)。改为
#   **首次调用时**函数内延迟导入(手法同 N-09:vllm/sm70_qwen38_ptr_fastpath.py 的
#   _HOST_SYMBOLS 缓存)。调用 `process_weights_after_loading` 时宿主必然已定义完毕,
#   行为逐位等价、去掉隐式顺序依赖。
_HOST_SYMBOLS = None


def _flash_loading_host_symbols():
    global _HOST_SYMBOLS
    if _HOST_SYMBOLS is None:
        from ..nvfp4_sm70_moe import (
            _COMPACT_GROUPED_MAX_SLOTS,
            _GRAPH_SAFE_MAX_TOKENS,
            _get_qwen38_raw_scale_workspace,
            _raw_scales_match_prepared,
            _validate_weight_layout,
            validate_nvfp4_sm70_moe_contract,
        )
        _HOST_SYMBOLS = (
            _COMPACT_GROUPED_MAX_SLOTS,
            _GRAPH_SAFE_MAX_TOKENS,
            _get_qwen38_raw_scale_workspace,
            _raw_scales_match_prepared,
            _validate_weight_layout,
            validate_nvfp4_sm70_moe_contract,
        )
    return _HOST_SYMBOLS


from .weights import (
    build_hotcold_ptrs,
    fill_hotcold_w13,
    fill_hotcold_w2,
    finalize_hotcold,
    resolve_hotcold_state,
)

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")


def process_weights_after_loading(self, layer: RoutedExperts) -> None:
    (
        _COMPACT_GROUPED_MAX_SLOTS,
        _GRAPH_SAFE_MAX_TOKENS,
        _get_qwen38_raw_scale_workspace,
        _raw_scales_match_prepared,
        _validate_weight_layout,
        validate_nvfp4_sm70_moe_contract,
    ) = _flash_loading_host_symbols()
    required_ops = (
        "nvfp4_sm70_prepare",
        "nvfp4_moe_dense_stage_sm70_out",
        "awq_moe_build_strided_ptrs",
    )
    missing = [name for name in required_ops if not hasattr(torch.ops._C, name)]
    if (
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE
        or envs.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE
    ) and not sm70_ops.has_nvfp4_qpn_m1_dispatch():
        missing.append("nvfp4_moe_qpn_m1_sm70_out")
    w2_direct_reduce_requested = bool(
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE
    )
    w2_direct_reduce_available = sm70_ops.has_nvfp4_qwen38_w2_direct_reduce()
    w2_direct_reduce_explicit = (
        "VLLM_SM70_NVFP4_QWEN38_MOE_W2_DIRECT_REDUCE" in os.environ
    )
    if (
        w2_direct_reduce_requested
        and not w2_direct_reduce_available
        and w2_direct_reduce_explicit
    ):
        missing.append("nvfp4_qwen38_w2_direct_reduce_out")
    elif w2_direct_reduce_requested and not w2_direct_reduce_available:
        logger.warning_once(
            "The default SM70 Qwen3.8 W2 direct-reduce op is absent from "
            "the loaded extension; retaining separate W2 and weighted "
            "reduce kernels. Explicit opt-in fails closed."
        )
    if (
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE
        and not sm70_ops.has_nvfp4_qpn_mtp5_dispatch()
    ):
        missing.append("nvfp4_moe_qpn_mtp5_sm70_out")
    indexed_prefill_ops = {
        "nvfp4_moe_indexed_dense_stage_sm70_out": hasattr(
            torch.ops._C, "nvfp4_moe_indexed_dense_stage_sm70_out"
        ),
        "moe_permute_metadata_with_scratch": hasattr(
            torch.ops._moe_C, "moe_permute_metadata_with_scratch"
        ),
    }
    indexed_prefill_requested = bool(
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL
    )
    indexed_prefill_available = all(indexed_prefill_ops.values())
    indexed_prefill_explicit = (
        "VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL" in os.environ
    )
    if (
        indexed_prefill_requested
        and not indexed_prefill_available
        and indexed_prefill_explicit
    ):
        missing.extend(
            name for name, available in indexed_prefill_ops.items() if not available
        )
    elif indexed_prefill_requested and not indexed_prefill_available:
        logger.warning_once(
            "The default SM70 Qwen3.8 indexed-A prefill route is not "
            "present in the loaded extension; falling back to the "
            "materialized-input route. Explicitly setting "
            "VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL=1 fails closed."
        )
    if missing:
        raise RuntimeError(
            "SM70 NVFP4 MoE requires the TurboMind extension "
            "with " + ", ".join(missing) + "."
        )
    if not hasattr(torch.ops._moe_C, "moe_permute_with_scratch"):
        raise RuntimeError("SM70 NVFP4 MoE requires graph-safe MoE permute ops.")
    if self.moe.has_bias:
        raise NotImplementedError("SM70 NVFP4 MoE does not support expert bias.")
    if layer.activation != MoEActivation.SILU:
        raise NotImplementedError(
            "SM70 NVFP4 MoE currently supports SwiGLU/SILU only."
        )
    if layer.apply_router_weight_on_input:
        raise NotImplementedError(
            "SM70 NVFP4 MoE does not support router weights on input."
        )
    if layer.expert_map is not None:
        raise NotImplementedError(
            "SM70 NVFP4 MoE currently requires fully replicated experts."
        )
    if layer.local_num_experts != layer.global_num_experts:
        raise NotImplementedError(
            "SM70 NVFP4 MoE currently requires local and global experts to match."
        )

    validate_nvfp4_sm70_moe_contract(layer.moe_config)
    _validate_weight_layout(layer)
    num_experts = int(layer.local_num_experts)
    hidden = int(layer.moe_config.hidden_dim)
    intermediate = int(layer.moe_config.intermediate_size_per_partition)
    fused_swiglu_requested = bool(
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL
        and int(layer.moe_config.tp_size) == 4
        and num_experts == 512
        and hidden == 2560
        and intermediate == 160
        and int(layer.moe_config.experts_per_token) == 10
        and layer.swiglu_limit is None
    )
    fused_swiglu_available = hasattr(
        torch.ops._C, "nvfp4_moe_indexed_fused_swiglu_sm70_out"
    )
    fused_swiglu_explicit = (
        "VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL" in os.environ
    )
    if (
        fused_swiglu_requested
        and not fused_swiglu_available
        and fused_swiglu_explicit
    ):
        raise RuntimeError(
            "SM70 Qwen3.8 fused-SwiGLU prefill requires the TurboMind "
            "extension with nvfp4_moe_indexed_fused_swiglu_sm70_out."
        )
    if fused_swiglu_requested and not fused_swiglu_available:
        logger.warning_once(
            "The default SM70 Qwen3.8 fused-SwiGLU prefill op is absent "
            "from the loaded extension; retaining the standalone "
            "activation route. Explicit opt-in fails closed."
        )
    fused_swiglu_prefill = bool(fused_swiglu_requested and fused_swiglu_available)
    fused_swiglu_decode = bool(
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE
        and fused_swiglu_prefill
        and sm70_ops.has_nvfp4_qwen38_w13_fused_swiglu()
    )
    if (
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE
        and fused_swiglu_prefill
        and not fused_swiglu_decode
    ):
        logger.warning_once(
            "The SM70 Qwen3.8 fused W13/SwiGLU decode op is absent; "
            "retaining separate exact W13 and activation kernels."
        )
    fast_prefill = bool(
        fused_swiglu_prefill and envs.VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL
    )
    raw_scale_requested = bool(
        envs.VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE
        and int(layer.moe_config.tp_size) == 4
        and num_experts == 512
        and hidden == 2560
        and intermediate == 160
        and int(layer.moe_config.experts_per_token) == 10
    )
    raw_scale_available = sm70_ops.has_nvfp4_qpn_raw_scale_dispatch()
    raw_scale_explicit = "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE" in os.environ
    if raw_scale_requested and not raw_scale_available and raw_scale_explicit:
        raise RuntimeError(
            "SM70 Qwen3.8 raw E4M3 scale storage requires the matching "
            "QPN decode and prefill-expansion operators."
        )
    if raw_scale_requested and not raw_scale_available:
        logger.warning_once(
            "The default SM70 Qwen3.8 raw E4M3 scale operators are not "
            "present; retaining persistent FP16 prepared scales. An "
            "explicit opt-in fails closed."
        )
    raw_scale = bool(raw_scale_requested and raw_scale_available)

    glm53_fused_permute_requested = bool(
        envs.VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8
        and int(layer.moe_config.tp_size) in (4, 8)
        and num_experts == 288
        and hidden == 4096
        and intermediate in (256, 512)
        and int(layer.moe_config.experts_per_token) == 8
    )
    glm53_fused_permute_available = hasattr(
        torch.ops._C, "sm70_glm53_moe_permute_q8_out"
    )
    glm53_fused_permute_explicit = (
        "VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8" in os.environ
    )
    if (
        glm53_fused_permute_requested
        and not glm53_fused_permute_available
        and glm53_fused_permute_explicit
    ):
        raise RuntimeError(
            "SM70 GLM-5.3 fused q8 MoE permute requires the TurboMind "
            "extension with sm70_glm53_moe_permute_q8_out."
        )
    if glm53_fused_permute_requested and not glm53_fused_permute_available:
        logger.warning_once(
            "The default SM70 GLM-5.3 fused q8 MoE permute op is absent "
            "from the loaded extension; retaining the generic graph-safe "
            "permute route. Explicit opt-in fails closed."
        )
    glm53_fused_permute_q8 = bool(
        glm53_fused_permute_requested and glm53_fused_permute_available
    )
    glm53_qpn_w13_requested = bool(
        envs.VLLM_SM70_GLM53_MOE_QPN_W13_Q8
        and glm53_fused_permute_q8
        and int(layer.moe_config.tp_size) == 8
        and num_experts == 288
        and hidden == 4096
        and intermediate == 256
        and int(layer.moe_config.experts_per_token) == 8
    )
    glm53_qpn_w13_available = hasattr(
        torch.ops._C, "nvfp4_glm53_moe_q8_qpn_sm70_out"
    )
    glm53_qpn_w13_explicit = "VLLM_SM70_GLM53_MOE_QPN_W13_Q8" in os.environ
    if (
        glm53_qpn_w13_requested
        and not glm53_qpn_w13_available
        and glm53_qpn_w13_explicit
    ):
        raise RuntimeError(
            "SM70 GLM-5.3 TP8 q8 W13 QPN requires the TurboMind extension "
            "with nvfp4_glm53_moe_q8_qpn_sm70_out."
        )
    if glm53_qpn_w13_requested and not glm53_qpn_w13_available:
        logger.warning_once(
            "The default SM70 GLM-5.3 TP8 q8 W13 QPN op is absent; "
            "retaining the exact TurboMind split-3 path. Explicit opt-in "
            "fails closed."
        )
    glm53_qpn_w13_q8 = bool(glm53_qpn_w13_requested and glm53_qpn_w13_available)

    w13_tm_weights: list[torch.Tensor] = []
    w13_tm_scales: list[torch.Tensor] = []
    w13_raw_scale_codes: list[torch.Tensor] = []
    w13_raw_global_scales: list[torch.Tensor] = []
    w13_meta: list[torch.Tensor] = []
    w2_tm_weights: list[torch.Tensor] = []
    w2_tm_scales: list[torch.Tensor] = []
    w2_raw_scale_codes: list[torch.Tensor] = []
    w2_raw_global_scales: list[torch.Tensor] = []
    w2_meta: list[torch.Tensor] = []
    state = resolve_hotcold_state(
        layer, num_experts, raw_scale, fast_prefill, w13_meta, w2_meta)
    raw_scale = state["raw_scale"]
    fast_prefill = state["fast_prefill"]
    w13_meta = state["w13_meta"]
    w2_meta = state["w2_meta"]
    for expert_id in ([] if state["stream_active"] else range(num_experts)):
        w13_packed = unpack_mxfp4_weight(layer.w13_weight[expert_id].data)
        w13_scales = layer.w13_weight_scale[expert_id].float().clone()
        w13_global = layer.w13_weight_scale_2[expert_id].float()
        w13_scales[:intermediate].mul_(w13_global[0])
        w13_scales[intermediate:].mul_(w13_global[1])
        prepared_w13 = sm70_ops.nvfp4_sm70_prepare(
            w13_packed,
            w13_scales.half().t().contiguous(),
            NVFP4_GROUP_SIZE,
            interleave_gated_silu=fused_swiglu_prefill,
        )
        if state["hotcold"]:
            fill_hotcold_w13(state, prepared_w13, expert_id)
        else:
            w13_tm_weights.append(prepared_w13[0])
            w13_tm_scales.append(prepared_w13[1])
        if raw_scale:
            if fused_swiglu_prefill:
                physical_global = w13_global.repeat(
                    intermediate // 32
                ).repeat_interleave(32)
            else:
                physical_global = w13_global.repeat_interleave(intermediate)
            w13_raw_scale_codes.append(
                (prepared_w13[1].float() / physical_global[None, :].float())
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .contiguous()
            )
            w13_raw_global_scales.append(w13_global.contiguous())
        w13_meta.append(prepared_w13[2])

        w2_packed = unpack_mxfp4_weight(layer.w2_weight[expert_id].data)
        w2_global = layer.w2_weight_scale_2[expert_id].float().reshape(())
        w2_scales = layer.w2_weight_scale[expert_id].float() * w2_global
        prepared_w2 = sm70_ops.nvfp4_sm70_prepare(
            w2_packed,
            w2_scales.half().t().contiguous(),
            NVFP4_GROUP_SIZE,
        )
        if state["hotcold"]:
            fill_hotcold_w2(state, prepared_w2, expert_id)
        else:
            w2_tm_weights.append(prepared_w2[0])
            w2_tm_scales.append(prepared_w2[1])
        if raw_scale:
            w2_raw_scale_codes.append(
                (prepared_w2[1].float() / w2_global)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .contiguous()
            )
            w2_raw_global_scales.append(w2_global.reshape(1))
        w2_meta.append(prepared_w2[2])

    if state["hotcold"]:
        prepared_w13_scales, prepared_w2_scales = finalize_hotcold(layer, state)
    else:
        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        prepared_w13_scales = torch.stack(w13_tm_scales)
        prepared_w2_scales = torch.stack(w2_tm_scales)
    if raw_scale:
        raw_w13_codes = torch.stack(w13_raw_scale_codes)
        raw_w13_globals = torch.stack(w13_raw_global_scales).float()
        raw_w2_codes = torch.stack(w2_raw_scale_codes)
        raw_w2_globals = torch.stack(w2_raw_global_scales).float()
        scale_workspace = _get_qwen38_raw_scale_workspace(
            layer.w13_tm_weight.device
        )
        w13_workspace = scale_workspace.view(512, 160, 320)
        w2_workspace = scale_workspace[: 512 * 10 * 2560].view(512, 10, 2560)
        w13_exact = _raw_scales_match_prepared(
            w13_workspace,
            raw_w13_codes,
            raw_w13_globals,
            prepared_w13_scales,
            fused_swiglu_prefill,
        )
        w2_exact = _raw_scales_match_prepared(
            w2_workspace,
            raw_w2_codes,
            raw_w2_globals,
            prepared_w2_scales,
            False,
        )
        if not (w13_exact and w2_exact):
            if raw_scale_explicit:
                raise RuntimeError(
                    "SM70 Qwen3.8 raw E4M3 scale reconstruction is not "
                    "bitwise equal to the prepared FP16 checkpoint scales."
                )
            logger.warning_once(
                "SM70 Qwen3.8 raw E4M3 scale reconstruction did not match "
                "the prepared checkpoint scales; retaining the FP16 scale "
                "representation."
            )
            raw_scale = False

    if raw_scale:
        layer.w13_raw_scale_codes = Parameter(raw_w13_codes, requires_grad=False)
        layer.w13_raw_global_scales = Parameter(
            raw_w13_globals, requires_grad=False
        )
        layer.w2_raw_scale_codes = Parameter(raw_w2_codes, requires_grad=False)
        layer.w2_raw_global_scales = Parameter(raw_w2_globals, requires_grad=False)
        layer.w13_tm_scales = w13_workspace
        layer.w2_tm_scales = w2_workspace
    else:
        layer.w13_tm_scales = Parameter(prepared_w13_scales, requires_grad=False)
        layer.w2_tm_scales = Parameter(prepared_w2_scales, requires_grad=False)

    w13_k_ld = int(w13_meta[0][0].item())
    w13_q_ld = int(w13_meta[0][1].item())
    w2_k_ld = int(w2_meta[0][0].item())
    w2_q_ld = int(w2_meta[0][1].item())
    if state["hotcold"]:
        w13_ptrs, w2_ptrs = build_hotcold_ptrs(
            layer, state, w13_k_ld, w13_q_ld, w2_k_ld, w2_q_ld)
    else:
        w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            w2_k_ld,
            w2_q_ld,
            num_experts,
        )
    if fast_prefill:
        # TurboMind packs this exact N320 weight as five contiguous N64
        # tiles per expert. The N256/N64 views therefore partition the
        # existing allocation without copying it. Scales remain a strided
        # N320 view, so both subprojections retain the original q_ld.
        w13_flat = layer.w13_tm_weight.view(num_experts, -1)
        head_words = hidden * 256 // 8
        w13_head_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            w13_flat[:, :head_words],
            layer.w13_tm_scales[:, :, :256],
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        w13_tail_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            w13_flat[:, head_words:],
            layer.w13_tm_scales[:, :, 256:],
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        layer.w13_head_strided_ptrs_w = Parameter(
            w13_head_ptrs[0], requires_grad=False
        )
        layer.w13_head_strided_ptrs_s = Parameter(
            w13_head_ptrs[1], requires_grad=False
        )
        layer.w13_tail_strided_ptrs_w = Parameter(
            w13_tail_ptrs[0], requires_grad=False
        )
        layer.w13_tail_strided_ptrs_s = Parameter(
            w13_tail_ptrs[1], requires_grad=False
        )
    layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
    layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
    layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
    layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)

    layer.sm70_nvfp4_moe = True
    layer.sm70_nvfp4_num_experts = num_experts
    layer.sm70_nvfp4_hidden_size = hidden
    layer.sm70_nvfp4_intermediate_size = intermediate
    layer.sm70_nvfp4_top_k = int(layer.moe_config.experts_per_token)
    layer.sm70_nvfp4_w13_k_dim = hidden
    layer.sm70_nvfp4_w13_n_dim = 2 * intermediate
    layer.sm70_nvfp4_w2_k_dim = intermediate
    layer.sm70_nvfp4_w2_n_dim = hidden
    layer.sm70_nvfp4_group_size = NVFP4_GROUP_SIZE
    layer.sm70_nvfp4_qwen38_indexed_prefill = bool(
        indexed_prefill_requested and indexed_prefill_available
    )
    layer.sm70_nvfp4_qwen38_fused_swiglu_prefill = fused_swiglu_prefill
    layer.sm70_nvfp4_qwen38_fused_swiglu_decode = fused_swiglu_decode
    layer.sm70_nvfp4_qwen38_fast_prefill = fast_prefill
    layer.sm70_nvfp4_qwen38_raw_scale = raw_scale
    layer.sm70_nvfp4_qwen38_w2_direct_reduce = bool(
        w2_direct_reduce_requested and w2_direct_reduce_available
    )

    layer.sm70_glm53_fused_permute_q8 = glm53_fused_permute_q8
    layer.sm70_glm53_qpn_w13_q8 = glm53_qpn_w13_q8
    layer.sm70_nvfp4_graph_safe_max_tokens = _GRAPH_SAFE_MAX_TOKENS
    layer.sm70_nvfp4_compact_grouped_max_slots = _COMPACT_GROUPED_MAX_SLOTS
    grouped_requested = bool(
        envs.VLLM_SM70_NVFP4_MOE_GROUPED_DECODE
        and (num_experts, hidden, intermediate, layer.sm70_nvfp4_top_k)
        == (512, 2560, 160, 10)
        and not raw_scale
        and layer.swiglu_limit is None
    )
    if grouped_requested and not sm70_ops.has_nvfp4_grouped_decode_dispatch():
        raise RuntimeError(
            "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE requires a matching native build."
        )
    layer.sm70_nvfp4_grouped_decode = grouped_requested
    # [vnext-F1] fail-closed: this loading snapshot predates upstream's
    # grouped_mtp5 branch (nvfp4_sm70_moe.py:1200/1206/1208/1214) and never sets
    # layer.sm70_nvfp4_grouped_mtp5. The upstream entry reads that attribute via
    # getattr(..., False) and would silently disable the TP4-only grouped-MTP5
    # path. Refuse to start with a silently disabled feature instead.
    if envs.VLLM_SM70_NVFP4_MOE_GROUPED_MTP5:
        raise RuntimeError(
            "VLLM_SM70_NVFP4_MOE_GROUPED_MTP5 is not supported by the "
            "flashnext_sm70 loading snapshot (grouped_mtp5 branch absent); "
            "refusing to start with a silently disabled feature."
        )
    if grouped_requested:
        # Layer-owned metadata; W2 reuses sorted_output. No process-global
        # cache or additional activation buffer inside graph capture.
        device = layer.w13_tm_weight.device
        layer._nvfp4_grouped_rows = torch.empty(
            160, 8, dtype=torch.int32, device=device
        )
        layer._nvfp4_grouped_experts = torch.empty(
            160, dtype=torch.int32, device=device
        )
        layer._nvfp4_grouped_sizes = torch.empty(
            160, dtype=torch.int32, device=device
        )
        layer._nvfp4_grouped_total = torch.empty(
            1, dtype=torch.int32, device=device
        )
    self._allocate_graph_safe_decode_buffers(layer)

    del layer.w13_weight
    del layer.w13_weight_scale
    del layer.w13_weight_scale_2
    del layer.w13_input_scale
    del layer.w2_weight
    del layer.w2_weight_scale
    del layer.w2_weight_scale_2
    del layer.w2_input_scale
    # Release unused conversion blocks between layers instead of retaining
    # their high-water mark throughout model loading. Live prepared weights
    # and inference buffers are unaffected; this is not an inference hook.
    torch.accelerator.empty_cache()
    logger.info_once(
        "SM70 ModelOpt NVFP4 TurboMind MoE path enabled "
        "(hidden=%d, local_intermediate=%d, local_experts=%d, top_k=%d, "
        "graph_safe_decode=B1-B%d, compact_grouped_decode<=%d routed rows).",
        hidden,
        intermediate,
        num_experts,
        layer.sm70_nvfp4_top_k,
        _GRAPH_SAFE_MAX_TOKENS,
        _COMPACT_GROUPED_MAX_SLOTS,
    )
    if raw_scale:
        logger.info_once(
            "SM70 Qwen3.8 raw E4M3 expert-scale storage enabled; "
            "generic and prefill routes share one FP16 expansion workspace."
        )
    if fused_swiglu_prefill:
        logger.info_once(
            "SM70 Qwen3.8 indexed-A fused-SwiGLU prefill candidate "
            "enabled (interleaved W13, exact FP16 epilogue arithmetic)."
        )
    if fused_swiglu_decode:
        logger.info_once(
            "SM70 Qwen3.8 fused W13/SwiGLU decode route enabled "
            "(split16, exact FP16 rounding and activation arithmetic)."
        )
    if fast_prefill:
        logger.info_once(
            "SM70 Qwen3.8 NVFP4 fast grouped prefill enabled "
            "(zero-copy W13 N256+N64, cached-B W2)."
        )
    if glm53_fused_permute_q8:
        logger.info_once(
            "SM70 GLM-5.3 exact fused q8 MoE permute enabled "
            "(M8/K8/E288 stable sort and materialized expert rows)."
        )
