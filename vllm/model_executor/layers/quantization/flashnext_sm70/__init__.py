# SPDX-License-Identifier: Apache-2.0
"""flashnext_sm70 — ds-cloud 两档存储补丁包(由 nvfp4_sm70_moe.py 抽出)。

依赖 DAG(单向无环):tier_swap(叶) · telemetry(→tier_swap) · hotcold(→telemetry)
· tier_stream(→hotcold,pinned_host) · weights(→hotcold,tier_stream,tier_swap,pinned_host)
· loading(→weights,宿主入口) · method(→telemetry,tier_stream,loading,宿主入口) · mtp_fp8_dispatch(叶)。
本 __init__ 负责:① 按依赖导入各模块;② 执行原文件里**导入期副作用**(atexit 注册 + 图外钩子安装)
——顺序与原单文件一致;③ 提供 `install_auxiliary()`(附线)。
主接线由宿主文件 `nvfp4_sm70_moe.py` 末尾文件内直连完成(旧的 `install()` 插件入口已取消):
本 `__init__` 顶层唯一导出 = `install_auxiliary()`(主接线类经 `flashnext_sm70.method` 子模块直取);历史「向入口重导出」面因无消费者已清理
(2026-10-02 审计回源核实:整棵 vllm 树无包外引用)。

命名约定(L2 例外登记,2026-10-10 审计):本包内 `_` 前缀名 = **补丁命名空间**标记
(供包内跨模块引用,如 `tier_swap._FLASH_TIER_SWAP_ON`) —— 对 AGENTS.md 准则⑧
「私有 `_` 只表内部用」的**显式例外**,以此处登记为准;不改名(大范围改名风险 > 收益)。
"""
from __future__ import annotations

import atexit
import os

from . import tier_swap, telemetry, hotcold, tier_stream  # noqa: F401  (按 DAG 顺序)
from . import weights  # noqa: F401  (依赖 hotcold/tier_stream/tier_swap,排最后)

# ---- 对外出口 ----
# 唯一对外出口:`install_auxiliary()`(宿主文件末尾显式调用)。历史上曾向入口重导出
# hotcold/telemetry/tier_stream/tier_swap/weights 的一批名字;2026-10-02 审计回源核实
# 「整棵 vllm 树无包外消费者」后清理(宿主改为文件内直连,不再依赖重导出)。
__all__ = ["install_auxiliary"]

# ---- 导入期副作用(原单文件位置:阶段2c 之后 + 阶段3 尾部)----
atexit.register(telemetry._flash_dump_routing)
atexit.register(telemetry._flash_flush_routing_seq)
atexit.register(telemetry._flash_hitstat_flush)
atexit.register(tier_swap._flash_swap_report)
# 图外触发钩子挂载(Batch5 V2.4-5):检查返回值 —— 若 tier swap+graph 已开却一个都没挂上,
# 换权 tick 永不触发 ⇒ N-06 默认 fail-closed 拒起(受限环境可 FLASH_NVFP4_HOOK_FATAL=0 降级)。


def _flash_hook_fatal_enabled() -> bool:
    """N-06:钩子未挂时是否 fail-closed(拒起)。默认开;FLASH_NVFP4_HOOK_FATAL=0 退回仅告警。"""
    return (os.getenv("FLASH_NVFP4_HOOK_FATAL", "1") or "1").strip().lower() \
        not in ("0", "false", "no")


_flash_hook_installed = tier_swap._flash_swap_install_graph_hook()
if (tier_swap._FLASH_TIER_SWAP_ON and tier_swap._FLASH_SWAP_GRAPH
        and not _flash_hook_installed):
    _hook_msg = (
        "FLASH-SWAP: ⚠️ 图外触发钩子未挂载(installed=0)而 tier swap+graph 已开 —— "
        "换权 tick 不会触发,请检查 runner 导入。targets=%s"
        % ["%s.%s" % (m, c) for m, c in tier_swap._FLASH_SWAP_TARGETS]
    )
    # N-06(2026-10-08):默认 fail-closed —— 钩子没挂上会让「换权 tick 永不触发」静默运行。
    #   降级(受限环境,如 CPU 单测因 vllm.config 循环导入挂不上钩子):
    #   FLASH_NVFP4_HOOK_FATAL=0 退回仅告警。
    if _flash_hook_fatal_enabled():
        raise RuntimeError(
            _hook_msg + ";拒绝启动(fail-closed)。受限环境可设 "
            "FLASH_NVFP4_HOOK_FATAL=0 退回仅告警")
    tier_swap.logger.error(_hook_msg)


def install_auxiliary() -> None:
    """附线接线(默认关):MTP FP8 分派。

    ⚠️ 主接线(把上游 `ModelOptNvFp4SM70MoEMethod` 换成本套实现)已**不再**在此完成:
    改由宿主文件 `nvfp4_sm70_moe.py` 末尾**文件内直连**(import 本包 + 重绑类名)。
    TP2 指针表快路(件③)也**已移出本包**:触发逻辑现居独立模块
    `vllm/sm70_qwen38_ptr_fastpath.py`(顶层);宿主文件 `nvfp4_sm70_moe.py` 只保留
    尾块 import + apply() 内两行调用,本包不再持有、也不再安装。
    本函数只负责默认关闭的 MTP 附线,由宿主文件末尾显式调用。
    """
    from vllm.logger import init_logger

    _log = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")

    # MTP 投机头(FP8_PB_WO)分派补丁:默认关,仅当 FLASH_NVFP4_MTP_FP8_DISPATCH=1 时接管。
    try:
        from . import mtp_fp8_dispatch as _mtp_fp8_dispatch

        _mtp_fp8_dispatch.install_mtp_fp8_dispatch()
    except Exception as _mtp_err:  # noqa: BLE001
        _log.warning("FLASH-MTP: 分派补丁装载失败(已忽略,不影响主流程):%r", _mtp_err)
