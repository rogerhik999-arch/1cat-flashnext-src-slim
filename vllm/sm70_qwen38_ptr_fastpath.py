# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm70_qwen38_ptr_fastpath — 件③ host 触发半:TP2 指针表单 token decode 快路

(ds-cloud 引擎修补 · 件③)

引擎侧 = csrc 三件套(声明 `csrc/ops.h` / 注册 `csrc/torch_bindings.cpp` /
定义 `csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu` 的算子
`nvfp4_moe_qpn_m1_ptr_sm70_out`,内核主体 `qpn_m1_sm70_body`)。

本模块只承载 host 侧「判定 + 自包含 forward」;指针表由件②(两档存储)在物化 /
换专家时生产与维护,本模块**只读消费**、从不回写。

接线:入口 `nvfp4_sm70_moe.py` 的 `apply()` 调用点保留 2 行守卫(读本模块两符号),
故上游文件只余「hook」级 件③ 足迹。
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")

__all__ = ["_flash_ptr_fastpath_admit", "_flash_ptr_fastpath_apply"]
# 命名约定(L2 例外登记,2026-10-10 审计):此处 `_` 前缀名经 `__all__` 作**公共 API** 导出
#   = 补丁命名空间约定(与 flashnext_sm70 包一致),对 AGENTS.md 准则⑧ 的显式例外。

# 宿主符号延迟导入(单一真相源仍在入口;此处仅引用,不复制)。
# N-09(2026-10-08):旧版在**模块导入期**从宿主反向 import ⇒ 与「宿主末尾 import 本模块」
#   形成 import 期双向依赖(靠导入顺序工作)。改为运行时首次调用时导入:调用 apply 时
#   宿主必然已定义完毕,行为等价、去掉隐式顺序依赖。
_HOST_SYMBOLS = None


def _flash_ptr_host_symbols():
    global _HOST_SYMBOLS
    if _HOST_SYMBOLS is None:
        from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
            _QWEN38_QPN_M1_W13_SPLIT_K,
            _QWEN38_QPN_M1_W2_SPLIT_K,
            _single_token_weighted_reduce,
            _use_qwen38_indexed_prefill,
        )
        _HOST_SYMBOLS = (
            _QWEN38_QPN_M1_W13_SPLIT_K, _QWEN38_QPN_M1_W2_SPLIT_K,
            _single_token_weighted_reduce, _use_qwen38_indexed_prefill,
        )
    return _HOST_SYMBOLS


_PTR_TABLE_NAMES = None


def _flash_ptr_table_names():
    """四张 strided 指针表名(单一真源:件② tier_swap._FLASH_SWAP_PTR_NAMES)。

    M1③(2026-10-10):此前本模块与件② 各自定义同一字符串元组(一概念两实现),
    改为**延迟导入**件② 常量(避免模块级跨包 import 触发 flashnext_sm70 包初始化的
    顺序问题;首次调用后缓存)。tier_swap 是件②「指针表产者」,为唯一真源。
    """
    global _PTR_TABLE_NAMES
    if _PTR_TABLE_NAMES is None:
        from vllm.model_executor.layers.quantization.flashnext_sm70.tier_swap import (
            _FLASH_SWAP_PTR_NAMES,
        )
        _PTR_TABLE_NAMES = _FLASH_SWAP_PTR_NAMES
    return _PTR_TABLE_NAMES


def _flash_ptr_tables_ok(layer, topk_ids) -> bool:
    """N-10 host 侧防御:指针表 dtype/连续性 + expert_id 值域(best-effort)。

    正常路径全过 ⇒ 行为不变;异常 ⇒ 返回 False,回退上游安全路径(不碰 native 内核)。
    expert_id 值域检查仅在**非捕获期**做(capture 期做 reduce+.item() 会破坏 CUDA graph)。
    env FLASH_PTR_STRICT(默认 1;=0 关闭本防御)。
    """
    strict = (os.getenv("FLASH_PTR_STRICT", "1") or "1").strip().lower() \
        not in ("0", "false", "no")
    if not strict:
        return True
    for name in _flash_ptr_table_names():
        table = getattr(layer, name, None)
        # 件②产出的指针表 = 每专家 16 字节的 uint8 缓冲(_flash_merge_strided_ptrs)。
        if table is None or table.dtype != torch.uint8 or not table.is_contiguous():
            logger.warning_once(
                "FLASH-PTR: 指针表 %s 非法(需 uint8/contiguous) -> 回退上游路径", name)
            return False
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return True
    except Exception:  # noqa: BLE001
        return True
    try:
        e = int(getattr(layer, "num_experts", 0)
                or (getattr(layer, "_flash_swap", None) or {}).get("num_experts", 0) or 0)
        if e > 0 and bool(((topk_ids < 0) | (topk_ids >= e)).any().item()):
            logger.warning_once(
                "FLASH-PTR: expert_id 越界(值域需 [0,%d)) -> 回退上游路径", e)
            return False
    except Exception:  # noqa: BLE001
        return True
    return True


def _flash_ptr_fastpath_admit(layer, x, topk_ids) -> bool:
    """命中条件:TP2 + 单 token 形状 + 件② 指针表就位 + ptr op 可用。判定无副作用、不抛。"""
    try:
        moe_config = layer.moe_config
        # L3(2026-10-10):单 token 形状由 moe_config 派生,不再硬编码魔法数。
        #   hidden_dim=2560 / experts_per_token=10 即现役 E512/K10 配置;
        #   任一缺失时回落原硬编码值,保持行为逐位等价。
        hidden = int(getattr(moe_config, "hidden_dim", 0) or 0) or 2560
        top_k = int(getattr(moe_config, "experts_per_token", 0) or 0) or 10
        return bool(
            int(moe_config.tp_size) == 2
            and tuple(x.shape) == (1, hidden)
            and x.dtype == torch.float16
            and x.is_contiguous()
            and tuple(topk_ids.shape) == (1, top_k)
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and x.is_cuda
            and topk_ids.is_cuda
            and getattr(layer, "w13_strided_ptrs_w", None) is not None
            and getattr(layer, "w13_strided_ptrs_s", None) is not None
            and getattr(layer, "w2_strided_ptrs_w", None) is not None
            and getattr(layer, "w2_strided_ptrs_s", None) is not None
            and hasattr(torch.ops._C, "nvfp4_moe_qpn_m1_ptr_sm70_out")
            and _flash_ptr_tables_ok(layer, topk_ids)
        )
    except Exception:  # noqa: BLE001
        return False


def _flash_ptr_fastpath_apply(self, layer, x, topk_weights, topk_ids):
    """自包含单 token forward(指针表 QPN-M1);语义 = 上游 direct_qpn_m1 路径。"""
    (w13_split_k, w2_split_k, weighted_reduce,
     use_indexed) = _flash_ptr_host_symbols()
    logger.info_once(
        "FLASH-PTR: 件③ 指针表快路生效(TP2 单 token,E512/K10;读件②指针表;"
        f"splitk w13={w13_split_k} w2={w2_split_k})。"
    )
    indexed_w13 = use_indexed(layer, x, topk_ids)
    interleaved_w13 = bool(
        getattr(layer, "sm70_nvfp4_qwen38_fused_swiglu_prefill", False)
    )
    buffers = self._get_buffers(layer, 1, indexed_w13)
    output = buffers["output"]
    route_ids = topk_ids.view(-1)
    torch.ops._C.nvfp4_moe_qpn_m1_ptr_sm70_out(
        buffers["gate_up"], x, layer.w13_strided_ptrs_w, layer.w13_strided_ptrs_s,
        route_ids, True, w13_split_k,
    )
    self._apply_swiglu(
        layer, buffers["intermediate"], buffers["gate_up"], interleaved=interleaved_w13
    )
    torch.ops._C.nvfp4_moe_qpn_m1_ptr_sm70_out(
        buffers["sorted_output"], buffers["intermediate"],
        layer.w2_strided_ptrs_w, layer.w2_strided_ptrs_s,
        route_ids, False, w2_split_k,
    )
    weighted_reduce(buffers["sorted_output"], topk_weights, output)
    return output
