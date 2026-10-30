# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.method — 文件内直连的补丁子类

(ds-cloud 两档存储补丁)

`FlashNextNvFp4SM70MoEMethod` 继承上游 `ModelOptNvFp4SM70MoEMethod`,就地覆盖三处:

    create_weights                : super() 之后追加流式槽位安装(_flash_stream_install)
    process_weights_after_loading : 转交 loading.py(热/冷物化,含 5 钩子)
    apply                         : 追加路由记录

上游文件保持 1Cat 原版;由宿主文件 `nvfp4_sm70_moe.py` 末尾**文件内直连**接管
(modelopt.py 对上游类做函数内 lazy import,替换模块属性即生效)。
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
)

from . import loading
from .telemetry import _flash_record_routing
from .tier_stream import _FLASH_STREAM_ON, _FLASH_STREAM_FATAL, _flash_stream_install

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")


class FlashNextNvFp4SM70MoEMethod(ModelOptNvFp4SM70MoEMethod):
    """上游 sm70 NVFP4 MoE 方法 + ds-cloud 两档存储补丁。

    `class`:vLLM 量化方法接口要求(可插拔、每层实例),合规(AGENTS.md 准则 1)。
    """

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer, num_experts, hidden_size, intermediate_size_per_partition,
            params_dtype, **extra_weight_attrs,
        )
        if not _FLASH_STREAM_ON:
            return
        try:
            _flash_stream_install(layer, int(num_experts))
        except Exception as exc:  # noqa: BLE001
            # 流式是本部署的容量前提:安装失败后 6 个专家参数已被改成 meta
            # (tier_stream._flash_stream_install),继续 forward 会缺权重 ⇒ 默认 fail-closed
            # 中止启动;诊断场景可 FLASH_NVFP4_STREAM_FATAL=0 退回仅 warning。(Batch5 V2.4-3)
            if _FLASH_STREAM_FATAL:
                raise RuntimeError(
                    "FLASH-STREAM install failed (fail-closed): %r" % (exc,)) from exc
            logger.warning("FLASH-STREAM install failed(已忽略, FATAL=0): %r", exc)

    def process_weights_after_loading(self, layer) -> None:
        # 热/冷物化主路径(含 5 钩子);副本见 loading.py
        return loading.process_weights_after_loading(self, layer)

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        """入口 apply:前置路由记录(= 现役入口行为),其余原样交上游。"""
        _flash_record_routing(layer, topk_ids)
        return super().apply(
            layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )
