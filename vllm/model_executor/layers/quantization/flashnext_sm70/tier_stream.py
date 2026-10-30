# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.tier_stream — 两档流式加载 / 逐专家拦截

(ds-cloud 两档存储补丁)

在「权重加载阶段」就接管专家权重 —— 逐专家从 checkpoint 张量(引擎 mmap 懒读,
数据只是“路过”)现场做 TP 切分 + repack,写进预分配的**热(显存)/ 冷(pinned host)**
槽位,参数本体保持 meta 占位(零存储)。这样引擎不再需要物化 63.3GiB 原始专家、
也不再需要 UVAOffloader 往宿主搬 32GiB 冗余副本。默认关闭(FLASH_NVFP4_STREAM_LOAD=0)。
"""
from __future__ import annotations

import os
import torch
from vllm import _sm70_ops as sm70_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    unpack_mxfp4_weight,
)

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")  # 同一 logger 名,保持日志前缀一致

from .hotcold import _flash_resolve_hot_ids
from . import pinned_host

# =====================================================================================


# ================= FLASH PATCH: 两档流式加载 / 逐专家拦截 (ds-cloud 2026-09-26) =================
_FLASH_STREAM_ON = (os.getenv("FLASH_NVFP4_STREAM_LOAD", "0") or "0").strip().lower()\
    not in ("", "0", "false", "no", "off")
# 流式安装失败是否 fail-closed(默认开):安装失败后 6 个专家参数已改 meta,继续 forward 会缺权重,
# 故必须中止启动;诊断场景可 FLASH_NVFP4_STREAM_FATAL=0 退回仅 warning。(Batch5 V2.4-3)
_FLASH_STREAM_FATAL = (os.getenv("FLASH_NVFP4_STREAM_FATAL", "1") or "1").strip().lower()\
    not in ("", "0", "false", "no", "off")

# 每个专家需攒齐的 9 张(分量, shard_id):gate/up=w1/w3 的 weight/scale/scale_2,down=w2 的三张。
_FLASH_STREAM_NEEDED = 9
# 9 张的**精确键集**(base:shard)。判「攒齐」必须按集合,而非 len>=9 ——
# 否则重复/多余 tag 会凑足 9 个而触发残缺 finish(KeyError 或写坏槽)。
_FLASH_STREAM_EXPECTED = frozenset((
    "13w:w1", "13s:w1", "13s2:w1",
    "13w:w3", "13s:w3", "13s2:w3",
    "2w:w2", "2s:w2", "2s2:w2",
))

def _flash_stream_tp_size(layer):
    for path in (("moe_config", "moe_parallel_config", "tp_size"),
                 ("moe_config", "tp_size")):
        obj = layer
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return int(obj)
        except Exception:  # noqa: BLE001
            continue
    return 1


def _flash_stream_narrow(tensor, dim, tp_rank, tp_size):
    """照抄 layer.py `_load_w13`/`_load_w2` 的 TP 切分(逐 rank narrow)。"""
    if tp_size <= 1 or tensor.dim() == 0:
        return tensor
    per_rank = tensor.shape[dim] // tp_size
    return tensor.narrow(dim, per_rank * tp_rank, per_rank)


def _flash_stream_alloc_slots(count, prepared_w13, prepared_w2, device, pin=False):
    """按 (count,)+单专家形状分配 4 张槽(w13 权重/尺度 + w2 权重/尺度)。

    冷/pinned 档在 `FLASH_NVFP4_EXACT_HOST_ALLOC=1` 时走**精确字节**分配器
    (绕开 torch pinned 分配器的 2 幂取整,见 pinned_host);其余行为不变。
    """
    exact_cold = bool(pin) and pinned_host.enabled()

    def _slot(exemplar):
        if exact_cold:
            return pinned_host.alloc_pinned_tensor(
                count, tuple(exemplar.shape), exemplar.dtype)
        return torch.empty((count,) + tuple(exemplar.shape),
                           dtype=exemplar.dtype, device=device, pin_memory=pin)

    return {
        "w13": [_slot(prepared_w13[i]) for i in range(2)],
        "w2": [_slot(prepared_w2[i]) for i in range(2)],
    }


def _flash_stream_finish_expert(layer, state, expert_id, pending):
    """9 张齐 → 与现补丁完全相同的 repack 公式 → 写热/冷槽位。"""
    device = torch.cuda.current_device()
    inter = int(state["inter"])
    gate_weight = pending["13w:w1"]; gate_scale = pending["13s:w1"]; gate_scale_global = pending["13s2:w1"]
    up_weight = pending["13w:w3"]; up_scale = pending["13s:w3"]; up_scale_global = pending["13s2:w3"]
    down_weight = pending["2w:w2"]; down_scale = pending["2s:w2"]; down_scale_global = pending["2s2:w2"]
    w13_w = torch.cat([gate_weight, up_weight], dim=0).contiguous().to(device)
    w13_s = torch.cat([gate_scale, up_scale], dim=0).contiguous().to(device)
    w13_s2 = torch.stack([gate_scale_global.reshape(()), up_scale_global.reshape(())]).to(device)
    w2_weight = down_weight.contiguous().to(device)
    w2_scale = down_scale.contiguous().to(device)
    w2_s2 = down_scale_global.reshape(()).to(device)
    w13_packed = unpack_mxfp4_weight(w13_w)
    w13_scales = w13_s.float().clone()
    w13_global = w13_s2.float()
    w13_scales[:inter].mul_(w13_global[0])
    w13_scales[inter:].mul_(w13_global[1])
    prepared_w13 = sm70_ops.nvfp4_sm70_prepare(
        w13_packed, w13_scales.half().t().contiguous(), NVFP4_GROUP_SIZE,
        interleave_gated_silu=bool(state.get("fused_swiglu", False)),
    )
    w2_packed = unpack_mxfp4_weight(w2_weight)
    w2_global = w2_s2.float().reshape(())
    w2_scales = w2_scale.float() * w2_global
    prepared_w2 = sm70_ops.nvfp4_sm70_prepare(
        w2_packed, w2_scales.half().t().contiguous(), NVFP4_GROUP_SIZE,
    )
    if state["hot_slots"] is not None:
        # 不变式(4.7):槽位按首专家形状预建,后续专家形状必须一致,否则写入错位。
        _slot_shape = tuple(state["hot_slots"]["w13"][0].shape[1:])
        if tuple(prepared_w13[0].shape) != _slot_shape:
            raise RuntimeError(
                "FLASH-STREAM: 专家 %s 的 repack 形状 %s 与首专家槽 %s 不一致"
                % (expert_id, tuple(prepared_w13[0].shape), _slot_shape))
    if state["hot_slots"] is None:
        state["w13_meta"] = prepared_w13[2].clone()
        state["w2_meta"] = prepared_w2[2].clone()
        state["hot_slots"] = _flash_stream_alloc_slots(
            len(state["hot_ids"]), prepared_w13, prepared_w2, device)
        state["cold_slots"] = _flash_stream_alloc_slots(
            len(state["cold_ids"]), prepared_w13, prepared_w2, "cpu", pin=True)
    hot_index = state["hot_pos"].get(expert_id)
    if hot_index is not None:
        for i in range(2):
            state["hot_slots"]["w13"][i][hot_index].copy_(prepared_w13[i])
            state["hot_slots"]["w2"][i][hot_index].copy_(prepared_w2[i])
    else:
        # 两档:非热即冷(冷专家已在加载期 pinned host 落地)。
        cold_index = state["cold_pos"][expert_id]
        for i in range(2):
            state["cold_slots"]["w13"][i][cold_index].copy_(prepared_w13[i])
            state["cold_slots"]["w2"][i][cold_index].copy_(prepared_w2[i])
    state["count"] += 1


def _flash_stream_weight_loader(self, param, loaded_weight, weight_name, shard_id,
                                expert_id, return_success=False):
    """模块级拦截器:唯一对 experts 生效的钩子(见 E2 机制侦查)。"""
    state = getattr(self, "_flash_stream", None)
    if state is None:
        return False if return_success else None
    base = state["param_keys"].get(id(param))
    if base is None:
        # input_scale 等非 repack 参数:本方法 use_global_sf=False,直接忽略。
        return True if return_success else None
    tag = "%s:%s" % (base, shard_id)
    if base.endswith("s2"):
        piece = loaded_weight
    elif base.startswith("2"):
        piece = _flash_stream_narrow(loaded_weight, 1, state["tp_rank"], state["tp_size"])
    else:
        piece = _flash_stream_narrow(loaded_weight, 0, state["tp_rank"], state["tp_size"])
    pending = state["pending"].setdefault(expert_id, {})
    pending[tag] = piece
    if _FLASH_STREAM_EXPECTED.issubset(pending):
        _flash_stream_finish_expert(self, state, expert_id, pending)
        state["pending"].pop(expert_id, None)
    return True if return_success else None


def _flash_stream_install(layer, num_experts):
    hot = int(os.getenv("FLASH_NVFP4_HOT_EXPERTS", "0") or "0")
    layer_idx, ids, cold_ordered = _flash_resolve_hot_ids(layer, num_experts, hot)
    if ids:
        hot = len(ids)
    if not (0 < hot < num_experts):
        return
    hot_ids = list(ids) if ids else list(range(hot))
    hot_set = set(hot_ids)
    cold_ids = list(cold_ordered) if cold_ordered else [
        e for e in range(num_experts) if e not in hot_set]
    # 两档:非热专家**全部**入 pinned host(无「盘上冷尾」)。
    # 1) 重专家参数换成 meta 占位(形状/dtype 保留、零存储);必须在建 param_keys 之前换。
    for name in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2",
                 "w2_weight", "w2_weight_scale", "w2_weight_scale_2"):
        placeholder = getattr(layer, name)
        setattr(layer, name, torch.nn.Parameter(
            torch.empty_like(placeholder.data, device="meta"), requires_grad=False))
    state = {
        "hot_ids": hot_ids, "cold_ids": cold_ids,
        "hot_pos": {e: i for i, e in enumerate(hot_ids)},
        "cold_pos": {e: i for i, e in enumerate(cold_ids)},
        "num_experts": num_experts,
        "inter": int(layer.moe_config.intermediate_size_per_partition),
        "tp_rank": int(getattr(layer, "tp_rank", 0)),
        "tp_size": _flash_stream_tp_size(layer),
        "fused_swiglu": False,
        "pending": {}, "hot_slots": None, "cold_slots": None,
        "w13_meta": None, "w2_meta": None, "count": 0, "layer_idx": layer_idx,
        "param_keys": {
            id(getattr(layer, "w13_weight")): "13w",
            id(getattr(layer, "w13_weight_scale")): "13s",
            id(getattr(layer, "w13_weight_scale_2")): "13s2",
            id(getattr(layer, "w2_weight")): "2w",
            id(getattr(layer, "w2_weight_scale")): "2s",
            id(getattr(layer, "w2_weight_scale_2")): "2s2",
        },
    }
    layer._flash_stream = state
    import types
    layer.weight_loader = types.MethodType(_flash_stream_weight_loader, layer)
    logger.info(
        "FLASH-STREAM: layer=%d installed hot=%d cold=%d tp_rank=%d tp_size=%d",
        layer_idx, len(hot_ids), len(cold_ids),
        state["tp_rank"], state["tp_size"],
    )
# =====================================================================================
