# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.hotcold — 热/冷清单解析 + 层号口径 + 两档预算 + 指针合并

(ds-cloud 两档存储补丁;由 nvfp4_sm70_moe.py 机械抽出,行为不变)

依 AGENTS.md 准则②「共用层准入 = 同层 ≥2 个对象真用到,否则下沉到它唯一的用户」:
原 `layers.py`(`_flash_resolve_layer_index`)与 `budget.py`(预算三函数)此前**仅本模块**消费,
已**下沉回本对象**(2026-09-29 审计),行为不变。
"""
from __future__ import annotations

import os
import torch
from vllm.logger import init_logger

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")  # 同一 logger 名,保持日志前缀一致

from .telemetry import (
    _FLASH_LAYER_CALL,
    _flash_layer_index_from_name,
    _flash_load_expert_rank,
    _flash_load_hotlists,
)

# ---- 层号口径(原 layers.py,下沉)--------------------------------------------

def _flash_resolve_layer_index(layer):
    """解析层号(优先模块名 layers.N,否则用调用序号兜底)。

    M1(2026-10-10):`layers.N` 解析口径下沉到 telemetry._flash_layer_index_from_name
    单一真源;此处仅保留本对象特有的调用序兜底递增。
    """
    index = _flash_layer_index_from_name(layer)
    if index < 0:
        index = _FLASH_LAYER_CALL[0]
    _FLASH_LAYER_CALL[0] += 1
    return index

# ---- 两档预算(原 budget.py,下沉)--------------------------------------------

def _flash_env_expert_mib() -> float:
    """每专家显存额度(MiB);env 可覆盖,单一默认值来源(本模块内显存专家预算)。"""
    return float(os.getenv("FLASH_NVFP4_EXPERT_MIB", "1.4649") or "1.4649")


def _flash_env_num_layers() -> int:
    """MoE 层数;env 可覆盖,单一默认值来源(本模块内显存专家预算)。"""
    return int(os.getenv("FLASH_NVFP4_NUM_LAYERS", "48") or "48")


def _flash_expert_budget_mib():
    """预算式(util×total − 非专家 − 激活 − KV池)⇒ (budget_mib, ctx)。

    KV 优先:先从总显存扣 KV 预留,余量才给专家热槽。常数全部 env 可覆盖,
    便于用实跑校准。ctx 供日志 / 失败信息复用(N-08 显式档与 auto 档共用)。
    """
    # 常数说明(两处真相 = compose.yaml 显式声明;使用环境:TP2 / NVFP4+FP8 混搭 /
    #           PLE 走宿主卸载 / 本引擎版本 / max-num-batched-tokens 8192 / cudagraph [1,2])。
    # 实测交叉核对(G=192/224/256):加载 19.23/21.43/23.63 − 热槽 ⇒ 非专家≈6.05(恒定)。
    try:
        total_mib = torch.cuda.get_device_properties(
            torch.cuda.current_device()).total_memory / float(1024 ** 2)
    except Exception:  # noqa: BLE001
        total_mib = 32768.0
    util = float(os.getenv("GPU_MEM_UTIL", "0.95") or "0.95")
    nonexpert_gib = float(os.getenv("FLASH_NVFP4_NONEXPERT_GIB", "6.05") or "6.05")
    # 激活+图+workspace:基值 = 路数=1 实测;斜率已标定 ≈ 0(run_actcal.sh)。
    act_base_gib = float(os.getenv("FLASH_NVFP4_ACT_GIB", "2.67") or "2.67")
    act_per_lane_gib = float(
        os.getenv("FLASH_NVFP4_ACT_GIB_PER_LANE", "0.0") or "0.0")
    kv_bytes_per_token = float(
        os.getenv("FLASH_NVFP4_KV_BYTES_PER_TOKEN", "14124") or "14124")
    kv_eq_seqs = int(os.getenv("FLASH_NVFP4_KV_EQ_SEQS", "1") or "1")
    max_model_len = int(os.getenv("FLASH_NVFP4_MAX_MODEL_LEN", "262144") or "262144")
    # ── KV 池(显存预留;与「路数 / 最大上下文配置」解耦)────────────────────
    _kv_pool_raw = (os.getenv("FLASH_NVFP4_KV_POOL_GIB", "") or "").strip()
    _default_pool_gib = (262144.0 * kv_bytes_per_token) / float(1024 ** 3)
    try:
        kv_pool_gib = float(_kv_pool_raw) if _kv_pool_raw else _default_pool_gib
    except Exception:  # noqa: BLE001
        logger.warning("FLASH-BUDGET: KV_POOL_GIB=%r 无法解析,用默认 %.4fG",
                       _kv_pool_raw, _default_pool_gib)
        kv_pool_gib = _default_pool_gib
    kv_req_gib = (max_model_len * kv_bytes_per_token) / float(1024 ** 3)
    if kv_pool_gib < kv_req_gib:
        logger.warning(
            "FLASH-BUDGET: KV_POOL %.3fG < 单请求需求 %.3fG (mml=%d) -> 抬到需求值",
            kv_pool_gib, kv_req_gib, max_model_len)
        kv_pool_gib = kv_req_gib
    per_expert_mib = _flash_env_expert_mib()
    n_layers = _flash_env_num_layers()
    act_gib = act_base_gib + act_per_lane_gib * max(0, kv_eq_seqs - 1)
    budget_mib = (util * total_mib - nonexpert_gib * 1024.0 - act_gib * 1024.0
                  - kv_pool_gib * 1024.0)
    return budget_mib, {
        "util": util, "total_mib": total_mib, "nonexpert_gib": nonexpert_gib,
        "act_gib": act_gib, "act_per_lane_gib": act_per_lane_gib,
        "kv_pool_gib": kv_pool_gib, "per_expert_mib": per_expert_mib,
        "n_layers": n_layers, "max_model_len": max_model_len,
    }


def _flash_gpu_expert_budget_count(num_experts):
    """显存专家额度(每层热专家数)。KV 优先:先扣 KV 预留,余量才给专家。

    off(默认)/未设 -> None(沿用既有热集大小);auto -> 按公式算;<N> -> 直接取 N
    (N 亦做预算交叉核对,N-08)。常数全部可用 env 覆盖,便于用实跑校准。
    """
    raw = (os.getenv("FLASH_NVFP4_GPU_EXPERT_BUDGET", "") or "").strip().lower()
    if raw in ("", "off", "false", "no"):
        return None
    if raw != "auto":
        try:
            want = int(float(raw))
        except Exception:  # noqa: BLE001
            logger.warning("FLASH-BUDGET: GPU_EXPERT_BUDGET=%r 无法解析,忽略", raw)
            return None
        clamped = max(1, min(want, num_experts - 1))
        if clamped != want:
            logger.warning(
                "FLASH-BUDGET: GPU_EXPERT_BUDGET=%d 越界 -> 钳到 %d (有效范围 [1, %d])",
                want, clamped, num_experts - 1)
        # ── N-08(2026-10-08):显式档也与 KV/激活预算交叉核对(旧版只钳范围、不核预算)──
        budget_mib, ctx = _flash_expert_budget_mib()
        max_by_budget = int(budget_mib / (ctx["per_expert_mib"] * ctx["n_layers"]))
        if clamped > max_by_budget:
            _fatal = (os.getenv("FLASH_NVFP4_BUDGET_FATAL", "1") or "1").strip().lower() \
                not in ("0", "false", "no")
            _detail = (
                "FLASH-BUDGET: 显式额度 %d 超预算 max=%d "
                "(budget=%.1fMiB = util%.2f×%.0fMiB - 非专家%.2fG - 激活%.2fG - KV池%.3fG)"
                % (clamped, max_by_budget, budget_mib, ctx["util"], ctx["total_mib"],
                   ctx["nonexpert_gib"], ctx["act_gib"], ctx["kv_pool_gib"]))
            if _fatal:
                raise RuntimeError(
                    _detail + ";KV 优先预算不可满足,fail-closed 拒绝启动"
                    "(如需强行使用该值,设 FLASH_NVFP4_BUDGET_FATAL=0)")
            logger.warning(_detail + " -> 按显式值使用(FLASH_NVFP4_BUDGET_FATAL=0)")
        return clamped
    budget_mib, ctx = _flash_expert_budget_mib()
    per_expert_mib = ctx["per_expert_mib"]
    n_layers = ctx["n_layers"]
    floor = int(os.getenv("FLASH_NVFP4_GPU_EXPERT_FLOOR", "160") or "160")
    want = int(budget_mib / (per_expert_mib * n_layers))
    # V2.4-1 fail-closed(2026-10-08):budget 不足以支撑 floor(含 budget<0)⇒ **默认拒绝启动**,
    #   而非静默钳到 floor。理由:钳到 floor 会**超出**「KV 优先」预算、启动即埋 OOM 隐患,
    #   且与注释宗旨相悳(降级:env FLASH_NVFP4_BUDGET_FATAL=0 回旧行为「告警+钳 floor」)。
    if want < floor:
        _fatal = (os.getenv("FLASH_NVFP4_BUDGET_FATAL", "1") or "1").strip().lower() \
            not in ("0", "false", "no")
        _detail = (
            "FLASH-BUDGET: auto 额度不足 want=%d < floor=%d "
            "(budget=%.1fMiB = util%.2f×%.0fMiB - 非专家%.2fG - 激活%.2fG - KV池%.3fG)"
            % (want, floor, budget_mib, ctx["util"], ctx["total_mib"],
               ctx["nonexpert_gib"], ctx["act_gib"], ctx["kv_pool_gib"]))
        if _fatal:
            raise RuntimeError(
                _detail + ";KV 优先预算不可满足,fail-closed 拒绝启动"
                "(如需强行以 floor 兜底,设 FLASH_NVFP4_BUDGET_FATAL=0)")
        logger.warning(_detail + " -> 按 floor 下限兜底(FLASH_NVFP4_BUDGET_FATAL=0)")
        want = floor
    elif want > num_experts - 1:
        # 上限钳制:额度比专家数还大 ⇒ 用满全部专家即可,**良性**,不告警级升级。
        want = num_experts - 1
    logger.info(
        "FLASH-BUDGET: GPU 专家额度 auto -> G=%d/层 (budget=%.1fMiB util=%.2f mml=%d "
        "kvpool=%.3fG nex=%.2f act=%.2f act_per_lane=%.3f floor=%d)",
        want, budget_mib, ctx["util"], ctx["max_model_len"], ctx["kv_pool_gib"],
        ctx["nonexpert_gib"], ctx["act_gib"], ctx["act_per_lane_gib"], floor)
    return want


# 两档:冷专家额度不再按内存预算切分 —— 非热专家**全部** pinned host(UVA),
# 不存在「盘上冷尾」,故不再需要内存额度函数(_flash_cold_mem_budget_experts 已删除)。
# =====================================================================================

# ---- 热/冷清单解析 + 指针合并 ------------------------------------------------

def _flash_merge_strided_ptrs(hot_buf, hot_ids, cold_buf, cold_ids, num_experts):
    """把 hot/cold 两段「每专家 16 字节」指针数组合并成 num_experts 项的混合数组。

    两档:热(显存) ∪ 冷(pinned host) = 全部专家 ⇒ 每行必被覆盖,无空槽。
    """
    assert hot_buf.numel() == len(hot_ids) * 16, (hot_buf.numel(), len(hot_ids))
    assert cold_buf.numel() == len(cold_ids) * 16, (cold_buf.numel(), len(cold_ids))
    assert len(hot_ids) + len(cold_ids) == num_experts, (len(hot_ids), len(cold_ids))
    device = hot_buf.device
    out = torch.empty(num_experts * 16, dtype=torch.uint8, device=device)
    view = out.view(num_experts, 16)
    if len(hot_ids):
        view[torch.tensor(hot_ids, dtype=torch.long, device=device)] = hot_buf.view(-1, 16)
    if len(cold_ids):
        view[torch.tensor(cold_ids, dtype=torch.long, device=device)] = cold_buf.view(-1, 16)
    return out
# =====================================================================================

def _flash_resolve_hot_ids(layer, num_experts, default_h):
    """返回 (层号, 热 id 有序列表或 None, 冷 id 有序列表或 None)。

    热/冷均按「实测频次降序」(需 FLASH_NVFP4_EXPERT_RANK);无全序表则退回旧行为
    (热集表 / 下标前 N)。热集大小可由 FLASH_NVFP4_GPU_EXPERT_BUDGET 覆盖。
    """
    table = _flash_load_hotlists()
    index = _flash_resolve_layer_index(layer)
    budget = _flash_gpu_expert_budget_count(num_experts)
    rank_map = _flash_load_expert_rank()
    order = None
    if rank_map and index in rank_map:
        order = [expert_id for expert_id in rank_map[index] if 0 <= expert_id < num_experts]
    if order:
        if budget is not None:
            want = min(budget, num_experts - 1)
        elif table and index in table:
            want = min(len(table[index]), len(order))
        elif default_h > 0:
            want = min(default_h, len(order))
        else:
            want = 0
        if 0 < want < num_experts:
            hot = order[:want]
            hot_set = set(hot)
            # 冷集按「0..num_experts 全集补集」生成,与 hotlist 分支口径一致:
            # 即使 rank 表含越界 id 被过滤(order 变短),也保证两档并集 = 全部专家。
            return index, hot, [e for e in range(num_experts) if e not in hot_set]
        return index, None, None
    if table and index in table:
        hot = sorted({expert_id for expert_id in table[index] if 0 <= expert_id < num_experts})
        if budget is not None and 0 < budget < len(hot):
            logger.warning(
                "FLASH-BUDGET: 无专家全序表,预算 %d < 热集 %d -> 按 id 截断(非频次)",
                budget, len(hot))
            hot = hot[:budget]
        if 0 < len(hot) < num_experts:
            hot_set = set(hot)
            return index, hot, [expert_id for expert_id in range(num_experts) if expert_id not in hot_set]
        return index, None, None
    if budget is not None and 0 < budget < num_experts:
        return index, list(range(budget)), list(range(budget, num_experts))
    if not table:
        return index, None, None
    logger.warning("FLASH-HOTLIST: 无第 %d 层热集,退回下标前 %d", index, default_h)
    return index, None, None
