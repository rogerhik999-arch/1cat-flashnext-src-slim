# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.weights — 热/冷权重物化钩子(hot=显存 / cold=pinned host)

(ds-cloud 两档存储补丁;由 nvfp4_sm70_moe.py 的 process_weights_after_loading
内联段显式抽出,行为不变)

设计:上游 `process_weights_after_loading` 不改;我们只把「热/冷专家分档物化」
这一段(原内联 #4–#8)抽成具名钩子,以 `state`(dict)贯穿:

    resolve_hotcold_state   ← #4 前置:解析热/冷 id、诊断、流式槽位、swap 挂载
    fill_hotcold_w13 / _w2  ← #5/#6 循环体:逐专家填显存热缓冲 / pinned 冷缓冲
    finalize_hotcold        ← #7 循环后:落位 tm 权重、swap 预热、冷 UVA
    build_hotcold_ptrs      ← #8 分段建 strided 指针并 merge

不变式(I1/I5)在 `FLASH_NVFP4_INVARIANTS=1` 时于加载后校验(默认关,行为不变)。
"""
from __future__ import annotations

import os

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm.logger import init_logger

from .hotcold import _flash_merge_strided_ptrs, _flash_resolve_hot_ids
from .tier_stream import _FLASH_STREAM_ON
from . import pinned_host
from .tier_swap import (
    _FLASH_TIER_SWAP_ON,
    _FLASH_SWAP_GRAPH,
    _flash_swap_attach,
    _flash_swap_provision,
)

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")

# 加载期不变式开关(默认关 ⇒ 行为不变)
_FLASH_INVARIANTS_ON = (os.getenv("FLASH_NVFP4_INVARIANTS", "0") or "0").strip().lower()\
    in ("1", "true", "yes", "on")

_FLASH_DIAG_SEEN = [0]


def _flash_diag_enter(layer, num_experts, hotcold_h) -> None:
    """逐层打印显存/卸载/主机内存(只读观测,不改逻辑)。"""
    try:
        _FLASH_DIAG_SEEN[0] += 1
        _diag_seen = _FLASH_DIAG_SEEN[0]
        try:
            from vllm.model_executor.offloader.base import (
                get_offloader as _get_offloader,
            )
            _offloader = _get_offloader()
            _offload_bytes = getattr(_offloader, "cpu_offload_bytes", -1) / float(1024 ** 3)
            _offload_max_bytes = getattr(_offloader, "cpu_offload_max_bytes", -1) / float(1024 ** 3)
            _offload_name = type(_offloader).__name__
        except Exception as _e:  # noqa: BLE001
            _offload_bytes = _offload_max_bytes = -1.0
            _offload_name = "ERR:%r" % (_e,)
        try:
            _host_avail = -1.0
            with open("/proc/meminfo") as _file:
                for _line in _file:
                    if _line.startswith("MemAvailable"):
                        _host_avail = float(_line.split()[1]) / float(1024 ** 2)
                        break
        except Exception:  # noqa: BLE001
            _host_avail = -1.0
        _resident = 0
        for _param in layer.parameters():
            if not getattr(_param, "_vllm_is_uva_offloaded", False):
                _resident += _param.numel() * _param.element_size()
        logger.info(
            "FLASH-DIAG enter#%d experts=%d hot=%d "
            "cuda_alloc=%.2fMiB cuda_resv=%.2fMiB layer_gpu=%.2fMiB "
            "hostAvail=%.1fGiB offl=%.2f/%.2fGiB offloader=%s",
            _diag_seen, num_experts, hotcold_h,
            torch.cuda.memory_allocated() / 2 ** 20,
            torch.cuda.memory_reserved() / 2 ** 20,
            _resident / 2 ** 20, _host_avail, _offload_bytes, _offload_max_bytes, _offload_name,
        )
    except Exception as _e:  # noqa: BLE001
        logger.warning("FLASH-DIAG log failed: %r", _e)


def _flash_diag_exit() -> None:
    try:
        logger.info(
            "FLASH-DIAG exit#%d cuda_alloc=%.2fMiB cuda_resv=%.2fMiB",
            _FLASH_DIAG_SEEN[0] if _FLASH_DIAG_SEEN[0] else -1,
            torch.cuda.memory_allocated() / 2 ** 20,
            torch.cuda.memory_reserved() / 2 ** 20,
        )
    except Exception:  # noqa: BLE001
        pass


def resolve_hotcold_state(
    layer, num_experts, raw_scale, fast_prefill, w13_meta, w2_meta
) -> dict:
    """#4 前置:解析热/冷专家分档,返回贯穿后续钩子的 `state`。"""
    hotcold_h = int(os.getenv("FLASH_NVFP4_HOT_EXPERTS", "0") or "0")
    # 阶段2b:若提供逐层热集表/专家全序表,则按「逐层真实频次」选热(而非按专家下标)
    layer_idx, hot_ids_pref, cold_ordered = _flash_resolve_hot_ids(
        layer, num_experts, hotcold_h
    )
    if hot_ids_pref:
        hotcold_h = len(hot_ids_pref)
    hotcold = 0 < hotcold_h < num_experts
    hot_ids = list(hot_ids_pref) if hot_ids_pref else list(range(hotcold_h))
    hot_pos = {e: i for i, e in enumerate(hot_ids)}
    hot_set = set(hot_ids)
    cold_ids = list(cold_ordered) if cold_ordered else [
        e for e in range(num_experts) if e not in hot_set]
    cold_pos = {e: i for i, e in enumerate(cold_ids)}
    first_cold = cold_ids[0] if cold_ids else -1
    state = {
        "num_experts": num_experts,
        "layer_idx": layer_idx,
        "hotcold_h": hotcold_h,
        "hotcold": hotcold,
        "hot_ids": hot_ids,
        "hot_pos": hot_pos,
        "cold_ids": cold_ids,
        "cold_pos": cold_pos,
        "first_cold": first_cold,
        "stream_active": False,
        "w13_meta": w13_meta,
        "w2_meta": w2_meta,
        "raw_scale": raw_scale,
        "fast_prefill": fast_prefill,
        # 热:显存预分配缓冲
        "hot_w13": None, "hot_w13_scales": None, "hot_w2": None, "hot_w2_scales": None,
        # 冷:pinned CPU 预分配缓冲
        "cold_w13": None, "cold_w13_scales": None, "cold_w2": None, "cold_w2_scales": None,
    }
    # E2 流式:专家已在加载期由模块级拦截器 repack 进槽位;冷专家持真权重(pinned)。
    state["stream_active"] = bool(
        _FLASH_STREAM_ON
        and hotcold
        and getattr(layer, "_flash_stream", None) is not None
    )
    if hotcold:
        # 这些路径需要「整层连续大张量」或额外的 512-布局工作区,与热/冷不兼容
        if "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE" in os.environ:
            # 上游契约「显式即 fail-closed」:显式开了 RAW_SCALE 却又落进两档,不得静默吞掉
            # (否则 raw_scale 被回抄成 False、用户以为开了其实没开)。此处直接失败。
            raise RuntimeError(
                "FLASH: 两档(hotcold)与 VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE 不兼容;"
                "请勿在开启两档的同时显式设置该 opt-in(已按 fail-closed 处理)。")
        state["raw_scale"] = False
        state["fast_prefill"] = False
        _flash_diag_enter(layer, num_experts, hotcold_h)
    if state["stream_active"]:
        flash_state = layer._flash_stream
        state["hot_w13"] = flash_state["hot_slots"]["w13"][0]
        state["hot_w13_scales"] = flash_state["hot_slots"]["w13"][1]
        state["hot_w2"] = flash_state["hot_slots"]["w2"][0]
        state["hot_w2_scales"] = flash_state["hot_slots"]["w2"][1]
        state["cold_w13"] = flash_state["cold_slots"]["w13"][0]
        state["cold_w13_scales"] = flash_state["cold_slots"]["w13"][1]
        state["cold_w2"] = flash_state["cold_slots"]["w2"][0]
        state["cold_w2_scales"] = flash_state["cold_slots"]["w2"][1]
        state["hotcold_h"] = len(flash_state["hot_ids"])
        state["hot_ids"] = list(flash_state["hot_ids"])
        state["cold_ids"] = list(flash_state["cold_ids"])
        state["hot_pos"] = dict(flash_state["hot_pos"])
        state["cold_pos"] = dict(flash_state["cold_pos"])
        state["first_cold"] = state["cold_ids"][0] if state["cold_ids"] else -1
        # 两档:挂载动态换入换出(显存<->内存)
        if _FLASH_TIER_SWAP_ON:
            _flash_swap_attach(layer, num_experts)
            if getattr(layer, "_flash_swap", None) is not None:
                layer._flash_swap["layer_idx"] = layer_idx
                logger.info(
                    "FLASH-SWAP: layer=%d 动态换入换出已挂载(E=%d)",
                    layer_idx, num_experts)
        state["w13_meta"] = [flash_state["w13_meta"]]
        state["w2_meta"] = [flash_state["w2_meta"]]
        logger.info(
            "FLASH-STREAM: layer=%d 采用流式槽位 hot=%d cold=%d repacked=%d",
            flash_state["layer_idx"], len(state["hot_ids"]), len(state["cold_ids"]),
            flash_state["count"],
        )
    return state


def _flash_alloc_expert_slots(count, exemplar, device=None, pin_memory=False):
    """#5/#6 统一分配器：按 (count,)+单专家形状分配一张空缓冲。

    device 缺省沿用 exemplar 的 device；冷缓冲显式 device="cpu" + pinned。
    `FLASH_NVFP4_EXACT_HOST_ALLOC=1` 时 pinned 档走精确字节分配器(见 pinned_host)。
    """
    if pin_memory and pinned_host.enabled():
        return pinned_host.alloc_pinned_tensor(count, tuple(exemplar.shape), exemplar.dtype)
    return torch.empty(
        (count,) + tuple(exemplar.shape),
        dtype=exemplar.dtype,
        device=device if device is not None else exemplar.device,
        pin_memory=pin_memory,
    )


def fill_hotcold_w13(state: dict, prepared_w13, expert_id: int) -> None:
    """#5 循环体:把单专家 w13 填进显存热缓冲 / pinned 冷缓冲。"""
    hotcold_h = state["hotcold_h"]
    if state["hot_w13"] is None:
        # 惰性分配:由**首个进入热集**的专家触发。
        # (不能绑 expert_id==0:热集不含专家 0 时热缓冲永不分配,后续热专家 None[i] 即崩)
        state["hot_w13"] = _flash_alloc_expert_slots(hotcold_h, prepared_w13[0])
        state["hot_w13_scales"] = _flash_alloc_expert_slots(hotcold_h, prepared_w13[1])
    if expert_id in state["hot_pos"]:
        state["hot_w13"][state["hot_pos"][expert_id]].copy_(prepared_w13[0])
        state["hot_w13_scales"][state["hot_pos"][expert_id]].copy_(prepared_w13[1])
    else:
        if state["cold_w13"] is None:
            # 惰性分配:由**首个冷专家**触发(不再依赖 first_cold 恰好先到)
            state["cold_w13"] = _flash_alloc_expert_slots(
                state["num_experts"] - hotcold_h, prepared_w13[0], device="cpu", pin_memory=True)
            state["cold_w13_scales"] = _flash_alloc_expert_slots(
                state["num_experts"] - hotcold_h, prepared_w13[1], device="cpu", pin_memory=True)
        state["cold_w13"][state["cold_pos"][expert_id]].copy_(prepared_w13[0])
        state["cold_w13_scales"][state["cold_pos"][expert_id]].copy_(prepared_w13[1])


def fill_hotcold_w2(state: dict, prepared_w2, expert_id: int) -> None:
    """#6 循环体:把单专家 w2 填进显存热缓冲 / pinned 冷缓冲。"""
    hotcold_h = state["hotcold_h"]
    if state["hot_w2"] is None:
        # 惰性分配:由**首个进入热集**的专家触发(同 fill_hotcold_w13,不绑 expert_id==0)
        state["hot_w2"] = _flash_alloc_expert_slots(hotcold_h, prepared_w2[0])
        state["hot_w2_scales"] = _flash_alloc_expert_slots(hotcold_h, prepared_w2[1])
    if expert_id in state["hot_pos"]:
        state["hot_w2"][state["hot_pos"][expert_id]].copy_(prepared_w2[0])
        state["hot_w2_scales"][state["hot_pos"][expert_id]].copy_(prepared_w2[1])
    else:
        if state["cold_w2"] is None:
            state["cold_w2"] = _flash_alloc_expert_slots(
                state["num_experts"] - hotcold_h, prepared_w2[0], device="cpu", pin_memory=True)
            state["cold_w2_scales"] = _flash_alloc_expert_slots(
                state["num_experts"] - hotcold_h, prepared_w2[1], device="cpu", pin_memory=True)
        state["cold_w2"][state["cold_pos"][expert_id]].copy_(prepared_w2[0])
        state["cold_w2_scales"][state["cold_pos"][expert_id]].copy_(prepared_w2[1])


def finalize_hotcold(layer, state: dict):
    """#7 循环后:落位 tm 权重、swap 预热、冷 UVA。返回 (w13_scales, w2_scales)。"""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    num_experts = state["num_experts"]
    layer_idx = state["layer_idx"]
    # 加载完整性终检(流式路径):必须「全部专家到齐 + 无滞留 pending」。否则漏喂专家的
    # 槽位保持 torch.empty 随机位模式,运行期被路由命中即输出 NaN/垃圾且全程无告警。
    if state.get("stream_active"):
        _fs = getattr(layer, "_flash_stream", None) or {}
        _got, _pend = int(_fs.get("count", -1)), (_fs.get("pending") or {})
        if _got != num_experts or _pend:
            raise RuntimeError(
                "FLASH-STREAM: layer=%s 加载完整性终检失败——已 repack %d/%d 专家, 滞留 pending=%d"
                % (layer_idx, _got, num_experts, len(_pend)))
    hotcold_h = state["hotcold_h"]
    hot_ids = state["hot_ids"]
    cold_ids = state["cold_ids"]
    layer.w13_tm_weight = Parameter(state["hot_w13"], requires_grad=False)
    layer.w2_tm_weight = Parameter(state["hot_w2"], requires_grad=False)
    prepared_w13_scales = state["hot_w13_scales"]
    prepared_w2_scales = state["hot_w2_scales"]
    layer._flash_hotcold = True
    if _FLASH_TIER_SWAP_ON and _FLASH_SWAP_GRAPH:
        # 图外触发模式(SWAP_GRAPH=1)才需设备热度缓冲:必须在捕获前建好,否则首建
        # 落在捕获期会漏录。eager 档(图内触发)不用 heat_gpu ⇒ 不再白分配。
        _flash_swap_provision(layer, state["hot_w13"].device)
    layer._flash_hot_ids = list(hot_ids)
    layer._flash_cold_ids = list(cold_ids)
    # 单一真相源:让流式 state 的热/冷表**引用**运行期表(同一 list 对象);
    # 换档(swap)只原地改 layer._flash_hot_ids/_flash_cold_ids ⇒ state 不留陈旧副本。
    _flash_fs = getattr(layer, "_flash_stream", None)
    if _flash_fs is not None:
        _flash_fs["hot_ids"] = layer._flash_hot_ids
        _flash_fs["cold_ids"] = layer._flash_cold_ids
    logger.info(
        "FLASH-HOTLIST: layer=%d hot=%d cold=%d first_hot=%s",
        layer_idx, hotcold_h, len(cold_ids), hot_ids[:5],
    )
    # ★两档(显存 热 / 内存 冷):非热专家全部 pinned host 且 UVA 可见 ⇒ 被路由命中的
    #   专家**一律真算**(GPU 直读冷槽,不置 -inf、不改权重、不读零)。本补丁不含任何
    #   路由屏蔽(旧的 ROUTER_MASK 全冷屏蔽 / 三档盘上冷尾屏蔽 已随盘档一并删除)。
    layer._flash_cold_pinned = (
        state["cold_w13"], state["cold_w13_scales"], state["cold_w2"], state["cold_w2_scales"]
    )
    layer._flash_cold_uva = (
        get_accelerator_view_from_cpu_tensor(state["cold_w13"]),
        get_accelerator_view_from_cpu_tensor(state["cold_w13_scales"]),
        get_accelerator_view_from_cpu_tensor(state["cold_w2"]),
        get_accelerator_view_from_cpu_tensor(state["cold_w2_scales"]),
    )
    gib = sum(
        t.numel() * t.element_size()
        for t in (state["cold_w13"], state["cold_w13_scales"], state["cold_w2"], state["cold_w2_scales"])
    ) / float(1024 ** 3)
    logger.info(
        "FLASH-HOTCOLD: hot=%d cold=%d -> %s %.3f GiB",
        hotcold_h, num_experts - hotcold_h, "pinned host", gib,
    )
    _check_invariants_after_load(layer, state)
    _flash_diag_exit()
    return prepared_w13_scales, prepared_w2_scales


def build_hotcold_ptrs(layer, state: dict, w13_k_ld, w13_q_ld, w2_k_ld, w2_q_ld):
    """#8:分段建 strided 指针(热段 + 冷段),再按 expert 顺序 merge。返回 (w13_ptrs, w2_ptrs)。"""
    num_experts = state["num_experts"]
    hot_ids = layer._flash_hot_ids
    cold_ids = layer._flash_cold_ids
    uva_w13_weight, uva_w13_scale, uva_w2_weight, uva_w2_scale = layer._flash_cold_uva
    hw, hs = sm70_ops.awq_moe_build_strided_ptrs(
        layer.w13_tm_weight, layer.w13_tm_scales,
        w13_k_ld, w13_q_ld, len(hot_ids),
    )
    cw, cs = sm70_ops.awq_moe_build_strided_ptrs(
        uva_w13_weight, uva_w13_scale, w13_k_ld, w13_q_ld, len(cold_ids),
    )
    w13_ptrs = (
        _flash_merge_strided_ptrs(hw, hot_ids, cw, cold_ids, num_experts),
        _flash_merge_strided_ptrs(hs, hot_ids, cs, cold_ids, num_experts),
    )
    hw2, hs2 = sm70_ops.awq_moe_build_strided_ptrs(
        layer.w2_tm_weight, layer.w2_tm_scales,
        w2_k_ld, w2_q_ld, len(hot_ids),
    )
    cw2, cs2 = sm70_ops.awq_moe_build_strided_ptrs(
        uva_w2_weight, uva_w2_scale, w2_k_ld, w2_q_ld, len(cold_ids),
    )
    w2_ptrs = (
        _flash_merge_strided_ptrs(hw2, hot_ids, cw2, cold_ids, num_experts),
        _flash_merge_strided_ptrs(hs2, hot_ids, cs2, cold_ids, num_experts),
    )
    return w13_ptrs, w2_ptrs


def _check_invariants_after_load(layer, state: dict) -> None:
    """加载期不变式(I1/I5),仅 FLASH_NVFP4_INVARIANTS=1 时生效(默认关,行为不变)。"""
    if not _FLASH_INVARIANTS_ON:
        return
    num_experts = state["num_experts"]
    hot_ids = state["hot_ids"]
    cold_ids = state["cold_ids"]
    hot = set(hot_ids)
    cold = set(cold_ids)
    # I1 容量与互斥:热/冷各自无重复,且互斥、并集 = 全部专家
    assert len(hot) == len(hot_ids), "I1 热集有重复 id"
    assert len(cold) == len(cold_ids), "I1 冷集有重复 id"
    assert not (hot & cold), "I1 热/冷专家重叠"
    assert hot | cold == set(range(num_experts)), "I1 热∪冷未覆盖全部专家"
    logger.debug(
        "FLASH-INVARIANTS: layer=%d hot=%d cold=%d OK",
        state["layer_idx"], len(hot), len(cold),
    )
