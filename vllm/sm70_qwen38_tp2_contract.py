# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 Qwen3.8 TP2 解码契约开关(件⑤ 共用层,env 门控)。

单一真源:`VLLM_SM70_QWEN38_TP2_CONTRACT` 的真值判定**只此一处**。

上游把「quality-qualified SM70 Qwen3.8 解码快路」限定在 **TP4**;本开关(默认 **OFF**)
在 **TP2** 下放行同一契约,并接受 `modelopt_mixed` 量化。默认 OFF 时**字节等价**上游。

消费者(共 2 处 ⇒ 抽为共用层,依 AGENTS.md 准则 2;其中 **1 处直接 import,1 处传递消费**):
  · `vllm/config/vllm.py`                            —— 解码编译契约 + 拓扑默认开关
  · `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` —— FP16 行 GEMV 快路准入
    (传递消费:经 `config.vllm._is_sm70_qwen38_decode_compile_contract` 间接取用,非直接 import)

⚠ 2026-10-10 审计更正(M3):此前记「同层 2 个**直接**消费者」不确;实测整棵 vllm 树**仅 1 处直接 import** = `config/vllm.py:27`(用点 `:179`/`:240`)。保留为共用层而非下沉回 config 的理由:该「单点真值判定」被两处以不同路径使用,下沉会把判定逻辑绑死进 config 模块。
"""
from __future__ import annotations

import os

__all__ = ["tp2_contract_enabled"]

# 视作「关」的取值:与同批 flashnext_sm70 各门控统一为 5 词(含 "no"/"off"),额外保留 "none" 兼容。
# 其余(含 "1"/"true"/"yes" 等非空字符串)视作「开」。
_OFF_VALUES = ("", "0", "false", "no", "off", "none")


def tp2_contract_enabled() -> bool:
    """`VLLM_SM70_QWEN38_TP2_CONTRACT` 是否开启(未设 / "" / 0 / false / no / off / none ⇒ 关)。"""
    value = os.environ.get("VLLM_SM70_QWEN38_TP2_CONTRACT", "0").strip().lower()
    return value not in _OFF_VALUES
