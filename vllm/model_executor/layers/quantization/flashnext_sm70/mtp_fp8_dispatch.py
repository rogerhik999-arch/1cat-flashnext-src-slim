# SPDX-License-Identifier: Apache-2.0
"""flashnext_sm70.mtp_fp8_dispatch — 让 MTP 投机头(FP8 块量化专家)走 SM70 FP8 MoE 方法。

根因(2026-09-28 实测,**三处**):
 (1) 算法名无 MoE 实现:MTP 投机头专家层在 **config.json** 里 quant_algo=``FP8_PB_WO``
     (block 128, weight-only;检查点只有 ``weight``(F8_E4M3) + ``weight_scale_inv``(BF16)),
     而 ``ModelOptMixedPrecisionConfig.get_quant_method`` 只识别 ``FP8`` / ``NVFP4`` /
     ``W4A16_NVFP4`` ⇒ 落到 ``return None`` ⇒ MTP 专家退化为「未量化层」
     ⇒ 加载 ``...weight_scale_inv`` 时 ``getattr`` 失败:
        ``AttributeError: 'FusedMoE' object has no attribute 'w2_weight_scale_inv'``。
 (2) 前缀索引错位:MTP 层运行期前缀带 ``mtp_start_layer_idx`` 偏移 ——
     ``Qwen4ExpMultiTokenPredictor.mtp_start_layer_idx = config.num_hidden_layers``(=48),
     建层前缀 = ``f"mtp.layers.{48 + idx}"``(mtp.py:201/239)⇒ 实测 ``mtp.layers.48.mlp.experts``;
     而 config 键 = ``mtp.layers.0.mlp.experts`` ⇒ ``_resolve_quant_algo`` 直接返 None。
     故补丁必须先做 MTP 前缀回查(同后缀匹配),再判算法名。
 (3) 内核几何约束:SM70 FP8 MoE 内核 ``fp8_sm70_prepare`` 要求块 scale 严格 =
     ``[ceil(n/128), ceil(k/128)]``。MTP 专家中间维 = 640,TP=2 ⇒ 每卡分片 = 320,
     **320 不是 128 的倍数**:融合 w13 的 n = 640 ⇒ 内核要 5 块,而上游
     ``create_weights`` 给 ``2*ceil(320/128)=6`` ⇒ 6≠5,崩在权重后处理。
     修法(``Fp8SM70MoEPaddedMethod``):把每卡中间维 pad 到 128 对齐(320→384,
     整维 640→768),**权重补零、scale 补一块(=1.0)**,整维按 rank 均分 384 片;
     真实块的边界与检查点块边界完全重合 ⇒ 真实块 scale 原样落位(零块对结果无贡献)。

本补丁:
  门控(env ``FLASH_NVFP4_MTP_FP8_DISPATCH``,默认关)接管
  ``ModelOptMixedPrecisionConfig.get_quant_method`` —— 当层是 ``RoutedExperts``
  且解析出的算法名是 ``FP8_PB_WO`` 时,返回 ``Fp8SM70MoEPaddedMethod``
  (block 128、``weight_scale_inv`` 命名;按 128 对齐后再分片)。

纪律:
  默认关闭时行为与上游**完全一致**(零改动风险);开启时才接管。
  只读上游、不改上游文件。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")

# MTP 检查点的「块量化 weight-only」算法名(上游无 MoE 实现)。
# draft 路径实测读 config.json ⇒ 命中名 = FP8_PB_WO(容器内 resolve 实测值);
# hf_quant_config.json 另写 FP8_BLOCK_SCALES(主模型路径),一并收下以防其他加载路径。
_MTP_FP8_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")
# MTP 专家权重的块大小(检查点 group_size=128 ⇒ (128,128))。
_BLOCK_SIZE = (128, 128)

_installed = False


def _resolve_mtp_quant_algo(config, prefix: str) -> str | None:
    """运行期 MTP 前缀(带 num_hidden_layers 偏移)回查 config 键的算法名。

    运行期 = ``mtp.layers.<num_hidden_layers + idx>.<suffix>``,
    config 键 = ``mtp.layers.<idx>.<suffix>``。MTP 只有 1 层,故按「同后缀」回查即可;
    主模型前缀不以 ``mtp.layers.`` 开头 ⇒ 不会误伤。
    """
    if not prefix.startswith("mtp.layers."):
        return None
    body = prefix[len("mtp.layers.") :]
    if "." not in body:
        return None
    suffix = "." + body.split(".", 1)[1]
    quantized_layers = getattr(config, "quantized_layers", None) or {}
    for key, info in quantized_layers.items():
        if key.startswith("mtp.layers.") and key.endswith(suffix):
            algo = str(info.get("quant_algo", "")).upper()
            if algo:
                return algo
    return None


def _pad_up_to_block(size: int, block: int) -> int:
    """把 size 向上取整到 block 的整数倍。"""
    return ((size + block - 1) // block) * block


def _take_rank_shard(
    loaded_weight: torch.Tensor,
    dim: int,
    padded_full: int,
    per_rank: int,
    rank: int,
    pad_value: float,
) -> torch.Tensor:
    """沿 dim 把 loaded_weight pad 到 padded_full,再取第 rank 片(长 per_rank)。

    pad 追加在维尾部 ⇒ 真实块边界与检查点块边界(128)完全重合;
    补零块的 scale 任意(权重为零 ⇒ 对结果无贡献)。
    """
    current = loaded_weight.shape[dim]
    if current < padded_full:
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = padded_full - current
        pad = torch.full(
            pad_shape,
            pad_value,
            dtype=loaded_weight.dtype,
            device=loaded_weight.device,
        )
        loaded_weight = torch.cat([loaded_weight, pad], dim=dim)
    return loaded_weight.narrow(dim, per_rank * rank, per_rank)


def _install_padded_expert_loaders(layer, per_rank: int, block: int) -> None:
    """给 MTP 专家参数挂「整维 pad 到块对齐 → 再按 TP 分片」的 weight_loader。

    per_rank = 128 对齐后的每卡中间维(320→384);padded_full = per_rank*TP(768)。
    w13 融合维 = 2*per_rank(gate 片 ‖ up 片);w2 输入维 = per_rank。
    scale 的块维按 block 计:每卡 blocks_per_rank = per_rank/block。
    """
    tp_size = int(layer.tp_size)
    rank = int(layer.tp_rank)
    padded_full = per_rank * tp_size
    blocks_per_rank = per_rank // block
    blocks_full = blocks_per_rank * tp_size

    def _local_expert_id(expert_id: int) -> int:
        return layer._map_global_expert_id_to_local_expert_id(expert_id)

    def load_w13_weight(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        local_id = _local_expert_id(expert_id)
        if local_id == -1:
            return False if return_success else None
        piece = _take_rank_shard(loaded_weight, 0, padded_full, per_rank, rank, 0.0)
        base = 0 if shard_id == "w1" else per_rank
        param.data[local_id].narrow(0, base, per_rank).copy_(piece)
        return True if return_success else None

    def load_w13_scale(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        local_id = _local_expert_id(expert_id)
        if local_id == -1:
            return False if return_success else None
        piece = _take_rank_shard(
            loaded_weight, 0, blocks_full, blocks_per_rank, rank, 1.0
        )
        base = 0 if shard_id == "w1" else blocks_per_rank
        param.data[local_id].narrow(0, base, blocks_per_rank).copy_(piece)
        return True if return_success else None

    def load_w2_weight(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        local_id = _local_expert_id(expert_id)
        if local_id == -1:
            return False if return_success else None
        piece = _take_rank_shard(loaded_weight, 1, padded_full, per_rank, rank, 0.0)
        param.data[local_id].copy_(piece)
        return True if return_success else None

    def load_w2_scale(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        local_id = _local_expert_id(expert_id)
        if local_id == -1:
            return False if return_success else None
        piece = _take_rank_shard(
            loaded_weight, 1, blocks_full, blocks_per_rank, rank, 1.0
        )
        param.data[local_id].copy_(piece)
        return True if return_success else None

    for name, loader in (
        ("w13_weight", load_w13_weight),
        ("w13_weight_scale_inv", load_w13_scale),
        ("w2_weight", load_w2_weight),
        ("w2_weight_scale_inv", load_w2_scale),
    ):
        setattr(getattr(layer, name), "weight_loader", loader)


def _make_padded_fp8_sm70_moe_method():
    """构造 ``Fp8SM70MoEPaddedMethod``(延迟导入上游,避免默认关时也引入依赖)。"""
    from vllm.model_executor.layers.quantization.fp8_sm70_moe import (
        Fp8SM70MoEMethod,
    )

    class Fp8SM70MoEPaddedMethod(Fp8SM70MoEMethod):
        """MTP 专家几何修正:中间维按 128 对齐后再分片,满足内核块契约。

        ``create_weights`` 把每卡中间维 320 抬到 384(=128*3),相应把
        w13 融合维抬到 768、scale 块数抬到 6;并挂自定义 loader 做
        「整维 pad → 按 rank 分片」,真实块 scale 原样保留。

        `class`:vLLM 量化方法接口要求(可插拔),合规(AGENTS.md 准则 1)。
        """

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            block = int(self.weight_block_size[1])
            per_rank = _pad_up_to_block(intermediate_size_per_partition, block)
            super().create_weights(
                layer,
                num_experts,
                hidden_size,
                per_rank,
                params_dtype,
                **extra_weight_attrs,
            )
            _install_padded_expert_loaders(layer, per_rank, block)
            if per_rank != intermediate_size_per_partition:
                logger.info(
                    "FLASH-MTP: expert intermediate %d -> %d (block=%d aligned, "
                    "padded), fused w13 n=%d, w13 scale blocks=%d",
                    intermediate_size_per_partition,
                    per_rank,
                    block,
                    2 * per_rank,
                    2 * (per_rank // block),
                )

    return Fp8SM70MoEPaddedMethod


def _mtp_fp8_dispatch_enabled() -> bool:
    """门控开关;空/0/false/no/off 均视为关(默认关 ⇒ 与上游一致)。"""
    raw = (os.getenv("FLASH_NVFP4_MTP_FP8_DISPATCH", "0") or "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def install_mtp_fp8_dispatch() -> None:
    """接管 ModelOptMixedPrecisionConfig.get_quant_method(仅在门控开启时)。"""
    global _installed
    if _installed or not _mtp_fp8_dispatch_enabled():
        return

    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    original_get_quant_method = ModelOptMixedPrecisionConfig.get_quant_method
    padded_method_cls = _make_padded_fp8_sm70_moe_method()

    def get_quant_method(self, layer, prefix):  # type: ignore[no-untyped-def]
        if isinstance(layer, RoutedExperts):
            try:
                quant_algo = self._resolve_quant_algo(prefix)
            except Exception:  # noqa: BLE001 — 解析失败一律退回上游
                quant_algo = None
            if not quant_algo:
                # MTP 前缀带 +num_hidden_layers 偏移,上游直查落空 ⇒ 做同后缀回查。
                try:
                    quant_algo = _resolve_mtp_quant_algo(self, prefix)
                except Exception:  # noqa: BLE001
                    quant_algo = None
            if (quant_algo or "").upper() in _MTP_FP8_ALGOS:
                # Fp8SM70MoEMethod 只用 quant_config 的这两项:
                #   weight_block_size(块大小)、activation_scheme(是否静态激活量化)
                quant_config_view = SimpleNamespace(
                    weight_block_size=_BLOCK_SIZE,
                    activation_scheme="dynamic",  # weight-only ⇒ 非 static
                )
                logger.info(
                    "FLASH-MTP: prefix=%s quant_algo=%s -> Fp8SM70MoEPaddedMethod"
                    "(block=(128,128), weight-only, 128-aligned)",
                    prefix,
                    quant_algo,
                )
                return padded_method_cls(quant_config_view, layer)
        return original_get_quant_method(self, layer, prefix)

    ModelOptMixedPrecisionConfig.get_quant_method = get_quant_method
    _installed = True
    logger.info(
        "FLASH-MTP: FP8_PB_WO 分派补丁已装载(门控=开);MTP 专家改走 "
        "Fp8SM70MoEPaddedMethod(128 对齐)。"
    )
