# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.tier_swap — 两档换入换出(显存 hot / 宿主内存 cold)+ 图外触发钩子

(ds-cloud 两档存储补丁)

两级 = 显存(热)/ pinned 内存(冷)。
   铁律①:只一对相邻互换 —— 显存 ↔ 内存。
   铁律②:先换出再换入;淘汰对象 = 该档「热度末尾」。
   铁律③:热度阈值滞回(t1)。
   铁律④:周期重排(每 PERIOD 次层调用);R1 按 layer_idx 相位错开(stagger),
      把整周期搬运摊到多步(避免 48 层同相位 ⇒ 单步突发)。
   铁律⑤:命中即算 —— 冷专家本就 UVA 可见,GPU 直接读,不触发换。
机制:专家在显存/内存两档各有「定宽槽位」,内核按全局专家 id 读「strided 指针数组」
   (layer.*_strided_ptrs_{w,s},E×16 字节)。⇒ 换专家 = ① 1:1 搬数据(定宽 memcpy)
   ② 就地交换/改写指针数组条目(张量内容原地改 ⇒ CUDA graph 亦安全)。
依赖:FLASH_NVFP4_STREAM_LOAD=1 + FLASH_NVFP4_EXPERT_RANK。默认关闭。
⚠️ 仅当 FLASH_NVFP4_SWAP_GRAPH=0(旧「图内触发」)才需 --enforce-eager(CUDA graph 下
    每步 Python hook 不重放);=1(图外触发,现役默认)**无需** eager。
"""
from __future__ import annotations

import os
from contextlib import nullcontext

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")  # 同一 logger 名,保持日志前缀一致


# ============= FLASH PATCH: 两档动态换入换出(显存 热 / 内存 冷) =============
_FLASH_TIER_SWAP_ON = (os.getenv("FLASH_NVFP4_TIER_SWAP", "0") or "0").strip().lower()\
    not in ("", "0", "false", "no", "off")
# ── 换出/换入参数(源码默认 = 保守回退值;现役权威值以 compose.yaml 为准)──
# 注:与 compose 不同属**有意分层**(引擎侧保守 / compose 侧现役),见 容器设计笔记.md §八。
#     已登记分层(源↦现役):SWAP_PERIOD 64↦32 · SWAP_WARMUP 128↦16 · TIER_SWAP 0↦1 · SWAP_GRAPH 0↦1 · SWAP_PREFETCH 0↦1 · SWAP_STAGGER 0↦1。
#     改现役值请改 compose,并回头同步本注释,保持唯一真相。
_FLASH_SWAP_PERIOD = int(float(os.getenv("FLASH_NVFP4_SWAP_PERIOD", "64") or "64"))
_FLASH_SWAP_DECAY = float(os.getenv("FLASH_NVFP4_SWAP_DECAY", "0.99") or "0.99")
_FLASH_SWAP_T1 = float(os.getenv("FLASH_NVFP4_T1", "2") or "2")
_FLASH_SWAP_MAX = int(float(os.getenv("FLASH_NVFP4_SWAP_MAX", "4") or "4"))
_FLASH_SWAP_WARMUP = int(float(os.getenv("FLASH_NVFP4_SWAP_WARMUP", "128") or "128"))
# 换档"连续失败"熔断阈值:回滚成功但连续失败至此数 -> 冻结该层换权(默认 8;<=0 = **关闭熔断**,永不因失败计数冻结)。
_FLASH_SWAP_MAX_FAILS = int(float(os.getenv("FLASH_NVFP4_SWAP_MAX_FAILS", "8") or "8"))
# ── V2.2(2026-10-08):回滚失败(指针-数据可能错绑)的处置。默认 **fail-stop**
#    —— 立即终止进程,由容器 restart=unless-stopped 拉起干净实例。「宁可停服,不可错算」。
#    =0 退回旧行为(仅冻结该层),仅供 CPU 单测注入回滚失败用。
_FLASH_SWAP_FAILSTOP = (
    (os.getenv("FLASH_NVFP4_SWAP_FAILSTOP", "1") or "1").strip().lower()
    not in ("", "0", "false", "no", "off")
)
# 图外触发(FLASH_NVFP4_SWAP_GRAPH=1):热度在**设备侧**累积(可被 CUDA graph 捕获),
# 换权决策移到**图外**(GPUModelRunner.execute_model 头部,每步一次)。默认 0 = 旧行为(需 eager)。
_FLASH_SWAP_GRAPH = (os.getenv("FLASH_NVFP4_SWAP_GRAPH", "0") or "0").strip().lower()\
    not in ("", "0", "false", "no", "off")

# ── P2 预取换权(2026-09-30):把换权的 PCIe 搬运(D2H/H2D)放到**旁路 stream** 上,
#    与当步 compute 重叠;下一步开头 wait_event + **D2D 安装**(设备内,近免费)+ 翻转指针。
#    仅在图外触发(SWAP_GRAPH=1)下有效(否则无法推迟到「下一步开头」);
#    需额外暂存区(每层 ≤SWAP_MAX 专家;显存 + 锁页内存)。默认 0 = 关(纯同步,零行为变化)。
_FLASH_SWAP_PREFETCH = (
    (os.getenv("FLASH_NVFP4_SWAP_PREFETCH", "0") or "0").strip().lower()
    not in ("", "0", "false", "no", "off")
) and _FLASH_SWAP_GRAPH
_FLASH_SWAP_SIDE_STREAM = [None]

# 运行期不变式(I1 运行期 / I2 / I5)总闸:仅 FLASH_NVFP4_INVARIANTS=1 时校验(默认关 ⇒ 零行为变化)。
_FLASH_SWAP_INVARIANTS_ON = (
    (os.getenv("FLASH_NVFP4_INVARIANTS", "0") or "0").strip().lower()
    not in ("", "0", "false", "no", "off")
)
# ── R1(2026-09-30)相位错开(stagger):48 层原本**同相位**触发(每 PERIOD 步把整周期
#    搬运挤进「一步」⇒ 每 32 步一次的 ~73ms 突发,见 深挖-未解决的耗时-20260930.md)。
#    改按 layer_idx 相位偏移 ⇒ 每步只搬 ~48/PERIOD 层(~1.5 层 ≈13.7MiB ≈2.2ms),
#    藏进当步 compute。**触发机会不变(每层每周期恰一次;实际换权条数受 t1 采样相位影响可微差,见 AUDIT-R1)、不加显存、不改判据(t1/MAX/decay)。源码默认 0(保守回退);现役由 compose 给 1;=0 关(回原同相位)。**
_FLASH_SWAP_STAGGER = (
    (os.getenv("FLASH_NVFP4_SWAP_STAGGER", "0") or "0").strip().lower()
    not in ("", "0", "false", "no", "off")
)
_FLASH_SWAP_LAYERS: list = []      # attach 时登记,供图外钩子遍历
_FLASH_SWAP_STATS: dict = {
    "layers": 0, "steps": 0, "swap_t1": 0, "errors": 0, "rollback": 0,
}
_FLASH_SWAP_SLOT_NAMES = ("w13_tm_weight", "w13_tm_scales", "w2_tm_weight", "w2_tm_scales")
_FLASH_SWAP_PTR_NAMES = ("w13_strided_ptrs_w", "w13_strided_ptrs_s",
                         "w2_strided_ptrs_w", "w2_strided_ptrs_s")


def _flash_swap_get_slots(layer):
    """返回 (显存槽 4 元组, 内存槽 4 元组, 指针张量 4 元组)。"""
    gpu = tuple(getattr(layer, n, None) for n in _FLASH_SWAP_SLOT_NAMES)
    cold = getattr(layer, "_flash_cold_pinned", None) or (None, None, None, None)
    ptrs = tuple(getattr(layer, n, None) for n in _FLASH_SWAP_PTR_NAMES)
    return gpu, tuple(cold), ptrs


def _flash_swap_assert_state(layer) -> None:
    """运行期不变式(仅 `FLASH_NVFP4_INVARIANTS=1` 生效;默认关 ⇒ 零行为变化)。

    - **I1(运行期)/I5**:hot / cold 两档**互斥、并集 = 全部专家**(不丢、不重);
    - **I2(启发式)**:热/冷各专家的指针行两两不同 ⇒ 无「两专家共享同一行」的别名损坏。
    """
    if not _FLASH_SWAP_INVARIANTS_ON:
        return
    hot = [int(x) for x in (getattr(layer, "_flash_hot_ids", None) or [])]
    cold = [int(x) for x in (getattr(layer, "_flash_cold_ids", None) or [])]
    n = int((getattr(layer, "_flash_swap", None) or {}).get("num_experts", 0) or 0)
    assert len(set(hot)) == len(hot), "I1 热集重复"
    assert len(set(cold)) == len(cold), "I1 冷集重复"
    assert not (set(hot) & set(cold)), "I1 热/冷重叠"
    if n > 0:
        assert set(hot) | set(cold) == set(range(n)), "I5 两档并集 != 全部专家"
    _gpu, _cold, ptrs = _flash_swap_get_slots(layer)
    sig = {}
    for e in hot + cold:
        key = tuple(bytes(p.view(-1, 16)[e].tolist()) for p in ptrs if p is not None)
        assert key not in sig, "I2 指针行别名: 专家 %d 与 %d 共享" % (sig[key], e)
        sig[key] = e


def _flash_swap_attach(layer, num_experts=0) -> None:
    """加载期挂载:建每层热表(命中计数 + 衰减)。"""
    if getattr(layer, "_flash_swap", None) is not None:
        return
    count = int(num_experts) if num_experts else\
        int(getattr(layer, "sm70_nvfp4_num_experts", 0) or 0)
    if count <= 0:
        logger.warning("FLASH-SWAP: attach 跳过(num_experts=%s)", count)
        return
    layer._flash_swap = {
        "num_experts": count,
        "heat": torch.zeros(count, dtype=torch.float32),
        "last": torch.zeros(count, dtype=torch.int64),
        "t": 0,
        "calls": 0,
        "heat_gpu": None,
    }
    _FLASH_SWAP_LAYERS.append(layer)
    _FLASH_SWAP_STATS["layers"] += 1
    if not _FLASH_SWAP_GRAPH:
        # 图内触发模式需 eager;若开图且未设 SWAP_GRAPH=1,换入换出会静默为 0(实测)。
        logger.warning_once(
            "FLASH-SWAP: SWAP_GRAPH=0(图内触发)必须配 --enforce-eager;"
            "若开 CUDA graph 将 0 次换入换出。现役建议 FLASH_NVFP4_SWAP_GRAPH=1。"
        )


def _flash_swap_merge_counts(state, uniq, counts, decay) -> None:
    """按 (专家, 次数) 合并一步命中:先按「距上次命中步数」衰减,再累加。"""
    if uniq.numel() == 0:
        state["t"] += 1
        return
    age = (state["t"] - state["last"][uniq]).to(torch.float32)
    factor = torch.pow(torch.tensor(decay, dtype=torch.float32), age)
    state["heat"][uniq] = state["heat"][uniq] * factor + counts.to(torch.float32)
    state["last"][uniq] = state["t"]
    state["t"] += 1


def _flash_swap_bump(state, ids, decay) -> None:
    """命中计数:对选中的专家做「衰减后 +1」(CPU 路径,eager 用)。"""
    ids = ids.detach().to("cpu", dtype=torch.int64).reshape(-1)
    ids = ids[(ids >= 0) & (ids < state["num_experts"])]
    if ids.numel() == 0:
        state["t"] += 1
        return
    uniq, counts = torch.unique(ids, return_counts=True)
    _flash_swap_merge_counts(state, uniq, counts, decay)


def _flash_swap_effective_heat(state, decay):
    """当前有效热度(按「距上次命中」的步数衰减)。"""
    age = (state["t"] - state["last"]).to(torch.float32)
    return state["heat"] * torch.pow(torch.tensor(decay, dtype=torch.float32), age)


def _flash_swap_failstop(layer, cause, rollback_err, where) -> None:
    """V2.2:回滚失败(指针-数据可能错绑)的处置。默认 **fail-stop**(终止进程)。

    用 `os._exit` 而非抛异常:调用链 `_flash_swap_observe` 会吞掉换权异常(仅 warning)
    ⇒ 异常到不了「停止服务」;`os._exit` 才是「宁可停服、不可错算」的可靠语义。
    `FLASH_NVFP4_SWAP_FAILSTOP=0` 时退回旧行为(仅冻结该层,便于 CPU 单测注入)。
    """
    st = getattr(layer, "_flash_swap", None)
    if st is not None:
        st["frozen"] = True
    logger.critical(
        "FLASH-SWAP: %s 回滚失败 ⇒ 指针-数据可能错绑 ⇒ %s: %r / %r",
        where, "fail-stop" if _FLASH_SWAP_FAILSTOP else "仅冻结", cause, rollback_err)
    if not _FLASH_SWAP_FAILSTOP:
        logger.error("FLASH-SWAP: FLASH_NVFP4_SWAP_FAILSTOP=0 ⇒ 仅冻结该层(单测注入)")
        return
    try:
        import sys as _sys
        _sys.stdout.flush()
        _sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(1)


def _flash_swap_exchange(layer, gpu_index, host_index) -> None:
    """显存槽 gpu_index <-> 内存槽 host_index(1:1 互换;先换出再换入)。

    **单一事务(V2.2, 2026-10-08)**:把「数据段 + 四张指针表 + 热/冷 ID」当一个原子单元。
    变更前先取快照(数据 A/B 段、ID 旧值、指针表 a/b 行);任一阶段异常 ⇒ 用快照**全量回滚**
    (幂等);**回滚也失败 ⇒ fail-stop**(不只冻结)—— 消除旧版「元数据阶段在 try 之外」导致
    指针-数据**静默错绑**。
    """
    gpu, cold, ptrs = _flash_swap_get_slots(layer)
    hot_ids = layer._flash_hot_ids
    cold_ids = layer._flash_cold_ids
    a = int(hot_ids[gpu_index])
    b = int(cold_ids[host_index])
    with torch.no_grad():
        # 只读快照(不改任何槽):A 段来自显存槽,B 段来自内存槽
        a_snap = [gw[gpu_index].clone() for gw in gpu]
        b_snap = [hw[host_index].clone() for hw in cold]
        # 元数据快照:四张指针表第 a/b 行的旧值(供元数据阶段异常时回滚)
        ptr_rows = [(p, p.view(-1, 16)[a].clone(), p.view(-1, 16)[b].clone())
                    for p in ptrs if p is not None]

        def _restore_meta():
            # 幂等:若元数据尚未改,写回原值 = no-op;若半改,逐行还原
            for p, pa, pb in ptr_rows:
                view = p.view(-1, 16)
                view[a] = pa
                view[b] = pb
            hot_ids[gpu_index] = a
            cold_ids[host_index] = b

        def _restore_data():
            for gw, hw, a_s, b_s in zip(gpu, cold, a_snap, b_snap):
                hw[host_index].copy_(b_s)   # 内存槽写回原 B
                gw[gpu_index].copy_(a_s)    # 显存槽写回原 A

        try:
            for gw, hw, a_s, b_s in zip(gpu, cold, a_snap, b_snap):
                hw[host_index].copy_(a_s)   # 换出:A -> 内存
                gw[gpu_index].copy_(b_s)    # 换入:B -> 显存
            # 元数据阶段(指针 + 簿记)纳入同一事务
            for p, _pa, _pb in ptr_rows:
                view = p.view(-1, 16)
                keep = view[a].clone()
                view[a] = view[b]
                view[b] = keep
            hot_ids[gpu_index] = b
            cold_ids[host_index] = a
        except Exception as _e:  # noqa: BLE001
            try:
                _restore_meta()   # 幂等:覆盖三阶段(数据/指针/ID)
                _restore_data()
                _FLASH_SWAP_STATS["rollback"] += 1
                logger.warning("FLASH-SWAP: 换档中途异常已用快照回滚(swap 作废): %r", _e)
                # 回滚成功 ⇒ 数据无损,仅本次 swap 作废;连续失败超阈值才冻结
                st = getattr(layer, "_flash_swap", None)
                if st is not None:
                    st["fail_streak"] = int(st.get("fail_streak", 0)) + 1
                    if _FLASH_SWAP_MAX_FAILS > 0 and st["fail_streak"] >= _FLASH_SWAP_MAX_FAILS:
                        st["frozen"] = True
                        logger.error(
                            "FLASH-SWAP: 连续 %d 次换档失败 -> 冻结该层换权(避开持续无谓尝试/潜在错绑)",
                            st["fail_streak"])
            except Exception as _e2:  # noqa: BLE001
                _flash_swap_failstop(layer, _e, _e2, "同步换档")
            return  # 簿记保持不动 ⇒ 与「未交换」一致
    st = getattr(layer, "_flash_swap", None)
    if st is not None:
        st["fail_streak"] = 0
    _FLASH_SWAP_STATS["swap_t1"] += 1


def _flash_swap_side_stream():
    """旁路 stream(惰性建;无 CUDA 时返回 None ⇒ 退化为同步,便于 CPU 单测)。"""
    if not torch.cuda.is_available():
        return None
    if _FLASH_SWAP_SIDE_STREAM[0] is None:
        try:
            _FLASH_SWAP_SIDE_STREAM[0] = torch.cuda.Stream()
        except Exception as _e:  # noqa: BLE001
            logger.warning_once("FLASH-SWAP: 旁路 stream 创建失败(退回同步): %r", _e)
            return None
    return _FLASH_SWAP_SIDE_STREAM[0]


def _flash_swap_provision_staging(layer) -> None:
    """P2:为某层分配换权暂存区(显存 + 锁页内存);须在 CUDA graph 捕获前完成。失败退回同步。"""
    if not _FLASH_SWAP_PREFETCH:
        return
    # SWAP_MAX<=0 时暂存区只按 1 槽分配,而选对数上限是「不限」-> 会越界(审计 P2-1)。
    # 统一口径:不预取(回退同步路径,同步路径不索引暂存,行为与改前一致)。
    if _FLASH_SWAP_MAX <= 0:
        logger.warning_once(
            "FLASH-SWAP: SWAP_MAX=%d(<=0=不限)与预取暂存尺寸不兼容 -> 该层退回同步", _FLASH_SWAP_MAX)
        return
    state = getattr(layer, "_flash_swap", None)
    if state is None or state.get("staging_gpu") is not None:
        return
    gpu, cold, _ = _flash_swap_get_slots(layer)
    if any(x is None for x in gpu) or any(x is None for x in cold):
        # ★不得静默返回:曾因此让整个 P2 A/B 跑成废数据(槽未就绪没人喊)。
        _miss = ",".join([n for n, x in zip(_FLASH_SWAP_SLOT_NAMES, gpu) if x is None] +
                        ["cold_" + str(i) for i, x in enumerate(cold) if x is None])
        logger.warning_once(
            "FLASH-SWAP: 预取暂存区**未分配**(槽未就绪 layer=%s 缺=%s)-> 该层退回同步",
            state.get("layer_idx", "?"), _miss)
        return
    maxn = _FLASH_SWAP_MAX if _FLASH_SWAP_MAX > 0 else 1
    try:
        sg = [torch.empty((maxn,) + tuple(g.shape[1:]), dtype=g.dtype, device=g.device)
              for g in gpu]
        pin = torch.cuda.is_available()
        sh = [torch.empty((maxn,) + tuple(h.shape[1:]), dtype=h.dtype, pin_memory=pin)
              for h in cold]
        state["staging_gpu"] = sg
        state["staging_host"] = sh
        state["pending"] = []
        state["staging_event"] = None
        logger.info("FLASH-SWAP: 预取暂存区就绪(层=%s 槽=%d)",
                    state.get("layer_idx", "?"), maxn)
    except Exception as _e:  # noqa: BLE001
        state["staging_gpu"] = None
        state["staging_failed"] = True
        logger.warning_once("FLASH-SWAP: 预取暂存区分配失败(该层退回同步): %r", _e)


def _flash_swap_stage(layer, pairs) -> None:
    """P2 预取:旁路 stream 上把 A 段 D2H 到锁页暂存、B 段 H2D 到显存暂存(与当步 compute 重叠)。

    只**读**活动显存槽(与 compute 的读不冲突)、只**写**独立暂存缓冲 ⇒ 对当步前向零干扰;
    真正的安装(D2D + 指针翻转)推迟到下一步开头(_flash_swap_install)。
    """
    state = layer._flash_swap
    gpu, cold, _ = _flash_swap_get_slots(layer)
    sg = state["staging_gpu"]
    sh = state["staging_host"]
    hot_ids = layer._flash_hot_ids
    cold_ids = layer._flash_cold_ids
    stream = _flash_swap_side_stream()
    pending = []
    with torch.no_grad():
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            if stream is not None:
                # main -> side 依赖:旁路 stream 读活动槽前,必须等主流(含本步前向/上一步 install)
                # 对同槽的写。缺此依赖 = RAW/WAR 竞态(平时不触发,触发即数据↔簿记错绑)。
                stream.wait_stream(torch.cuda.current_stream())
            for k, (gi, hi) in enumerate(pairs):
                a = int(hot_ids[gi])
                b = int(cold_ids[hi])
                for j, (gw, hw) in enumerate(zip(gpu, cold)):
                    sh[j][k].copy_(gw[gi], non_blocking=(stream is not None))  # D2H:A->锁页暂存
                    sg[j][k].copy_(hw[hi], non_blocking=(stream is not None))  # H2D:B->显存暂存
                pending.append((k, gi, hi, a, b))
    ev = None
    if stream is not None:
        ev = torch.cuda.Event()
        ev.record(stream)
    state["pending"] = pending
    state["staging_event"] = ev
    if pending and not state.get("stage_logged"):
        state["stage_logged"] = True
        logger.info(
            "FLASH-SWAP: 首次预取搬运(层=%s 对数=%d stream=%s 段=%s)",
            state.get("layer_idx", "?"), len(pending),
            "async" if stream is not None else "sync-fallback",
            [(int(hot_ids[g]), int(cold_ids[h])) for (_k, g, h, _a, _b) in pending])


def _flash_swap_install(layer) -> None:
    """P2 安装:等旁路 stream 完成 -> D2D 安装(快) + 翻转指针/簿记(簿记只在安装成功后更新)。

    ⚠ L4-F5(审计·2026-10-03)时序前置(已登记,勿默默改动):
      本函数依赖「调用时刻当前 stream 上、上一步设备工作已排空」。该前置**非本函数自建**,
      由两处共同保证:① 上游 `_flash_swap_tick_graph` 在 install 之前的 `deltas=…to("cpu")`
      (**阻塞** D2H)已排空当前 stream;② 旁路预取另有 `staging_event` 显式 `wait_event` 兜底。
      ⇒ 若日后把 ① 改成 `non_blocking=True`/异步,隐式屏障会**静默消失**,须改为显式
      `wait_event(<上一步完成事件>)`。
    """
    state = layer._flash_swap
    pending = state.get("pending")
    if not pending:
        return
    gpu, cold, ptrs = _flash_swap_get_slots(layer)
    sg = state["staging_gpu"]
    sh = state["staging_host"]
    ev = state.get("staging_event")
    if ev is not None and torch.cuda.is_available():
        torch.cuda.current_stream().wait_event(ev)
    hot_ids = layer._flash_hot_ids
    cold_ids = layer._flash_cold_ids
    # 两阶段提交(V2.2, 2026-10-08):①先把数据搬到活动槽(数据层);②再翻指针+簿记(元数据层)。
    # 任一步异常都回滚(与同步路径 _flash_swap_exchange 同构),绝不静默错绑:
    #   数据恢复 = 用暂存(sh=锁存 A / sg=锁存 B)幂等覆盖回活动槽;
    #   元数据回滚 = 按**变更前登记**的日志(meta_journal)逐对还原 ⇒ **含当前半成品对**;
    #   回滚失败 ⇒ fail-stop(不只冻结)。
    done_data: list = []
    meta_journal: list = []   # [(gi, hi, a, b, ptr_rows)]:每对**变更前**登记 ⇒ 回滚必含当前对
    try:
        with torch.no_grad():
            for (k, gi, hi, a, b) in pending:
                done_data.append((k, gi, hi, a, b))  # 乐观登记(恢复幂等)
                for j, (gw, hw) in enumerate(zip(gpu, cold)):
                    gw[gi].copy_(sg[j][k])   # D2D 安装:B -> 活动显存槽
                    hw[hi].copy_(sh[j][k])   # A -> 活动冷水槽
            for (k, gi, hi, a, b) in pending:
                # 变更前登记该对元数据快照(ID 旧值 + 指针表 a/b 行)—— 回滚含**当前对**
                rows = [(p, p.view(-1, 16)[a].clone(), p.view(-1, 16)[b].clone())
                        for p in ptrs if p is not None]
                meta_journal.append((gi, hi, a, b, rows))
                hot_ids[gi] = b
                cold_ids[hi] = a
                for p in ptrs:
                    if p is None:
                        continue
                    view = p.view(-1, 16)
                    keep = view[a].clone()
                    view[a] = view[b]
                    view[b] = keep
    except Exception as _e:  # noqa: BLE001
        try:
            with torch.no_grad():
                for (gi, hi, a, b, rows) in reversed(meta_journal):   # 逐对还原(含当前对)
                    for p, pa, pb in rows:
                        view = p.view(-1, 16)
                        view[a] = pa
                        view[b] = pb
                    hot_ids[gi] = a
                    cold_ids[hi] = b
                for (k, gi, hi, a, b) in done_data:             # 暂存幂等恢复数据
                    for j, (gw, hw) in enumerate(zip(gpu, cold)):
                        gw[gi].copy_(sh[j][k])
                        hw[hi].copy_(sg[j][k])
            _FLASH_SWAP_STATS["rollback"] += 1
            logger.warning("FLASH-SWAP: 预取安装中途异常已回滚(swap 作废): %r", _e)
        except Exception as _e2:  # noqa: BLE001
            _flash_swap_failstop(layer, _e, _e2, "预取安装")
        state["pending"] = []
        state["staging_event"] = None
        return
    _FLASH_SWAP_STATS["swap_t1"] += len(pending)
    state["pending"] = []
    state["staging_event"] = None
    state["fail_streak"] = 0
    _FLASH_SWAP_STATS.setdefault("prefetch_installs", 0)
    _FLASH_SWAP_STATS["prefetch_installs"] += 1
    _flash_swap_assert_state(layer)
    if not state.get("install_logged"):
        state["install_logged"] = True
        logger.info("FLASH-SWAP: 首次预取安装完成(层=%s 安装对数=%d)",
                    state.get("layer_idx", "?"), len(pending))


def _flash_swap_step(layer) -> None:
    """周期重排:显存热 ↔ 内存冷(t1 阈值滞回;每次最多 SWAP_MAX 个)。"""
    state = getattr(layer, "_flash_swap", None)
    if state is None:
        return
    if state.get("frozen"):
        return  # 换档异常且回滚失败已冻结:停止该层换权,避免持续错绑
    _FLASH_SWAP_STATS["steps"] += 1
    _snapshot = _FLASH_SWAP_STATS["swap_t1"]
    heat = _flash_swap_effective_heat(state, _FLASH_SWAP_DECAY)
    hot_ids = layer._flash_hot_ids
    cold_ids = layer._flash_cold_ids
    cap = _FLASH_SWAP_MAX if _FLASH_SWAP_MAX > 0 else 10 ** 9
    # P2:预取暂存区**惰性分配** —— 首次 step 时所有槽必已就绪(否则无法交换);
    # 分配失败置 staging_failed 退回同步(不再重试),不影响换权本身。
    if _FLASH_SWAP_PREFETCH and state.get("staging_gpu") is None \
            and not state.get("staging_failed"):
        _flash_swap_provision_staging(layer)
    prefetch = (_FLASH_SWAP_PREFETCH and _FLASH_SWAP_MAX > 0
                and state.get("staging_gpu") is not None)
    staged_pairs = []
    chosen_gpu = set()
    chosen_host = set()
    for _ in range(cap):
        if not hot_ids or not cold_ids:
            break
        if prefetch:
            gi_choices = [i for i in range(len(hot_ids)) if i not in chosen_gpu]
            hi_choices = [i for i in range(len(cold_ids)) if i not in chosen_host]
            if not gi_choices or not hi_choices:
                break
            gpu_index = min(gi_choices, key=lambda i: float(heat[hot_ids[i]]))
            host_index = max(hi_choices, key=lambda i: float(heat[cold_ids[i]]))
        else:
            gpu_index = min(range(len(hot_ids)), key=lambda i: float(heat[hot_ids[i]]))
            host_index = max(range(len(cold_ids)), key=lambda i: float(heat[cold_ids[i]]))
        if float(heat[cold_ids[host_index]]) - float(heat[hot_ids[gpu_index]]) <= _FLASH_SWAP_T1:
            break
        if prefetch:
            # 预取::簿记此刻不动 ⇒ 用 excluded-set 防止本轮重复选同一对;安装时再统一更新。
            staged_pairs.append((gpu_index, host_index))
            chosen_gpu.add(gpu_index)
            chosen_host.add(host_index)
        else:
            _flash_swap_exchange(layer, gpu_index, host_index)
    if staged_pairs:
        _flash_swap_stage(layer, staged_pairs)
    now = _FLASH_SWAP_STATS["swap_t1"]
    if _FLASH_SWAP_STATS["steps"] % 100 == 0:
        logger.info(
            "FLASH-SWAP: 进度 steps=%d swap_t1=%d errors=%d",
            _FLASH_SWAP_STATS["steps"], now, _FLASH_SWAP_STATS["errors"])
    if now != _snapshot and not state.get("logged"):
        state["logged"] = True
        try:
            logger.info(
                "FLASH-SWAP: layer=%s 首次换入换出 -> G=%d M=%d (累积 swap_t1=%d) "
                "mem_heat_max=%.3f mem_heat_min=%.3f",
                state.get("layer_idx", "?"), len(hot_ids), len(cold_ids), now,
                float(heat[cold_ids].max()) if cold_ids else -1.0,
                float(heat[cold_ids].min()) if cold_ids else -1.0)
        except Exception as _e:  # noqa: BLE001
            logger.debug("FLASH-SWAP: 首次换档日志失败(忽略): %r", _e)
    _flash_swap_assert_state(layer)


def _flash_swap_phase_due(state) -> bool:
    """周期重排:本步是否该触发换权。

    R1:按 layer_idx 相位错开 —— 触发条件 (calls + layer_idx) % PERIOD == 0;
        48 层摊到 PERIOD 个相位,每步只 ~48/PERIOD 层触发(消单步突发)。
    STAGGER=0 → 退回原「同相位」:calls % PERIOD == 0(全层同时触发)。
    WARMUP 内不触发(不变)。
    """
    if _FLASH_SWAP_PERIOD <= 0:
        return False
    calls = int(state.get("calls", 0))
    if calls <= _FLASH_SWAP_WARMUP:
        return False
    if _FLASH_SWAP_STAGGER:
        offset = int(state.get("layer_idx", 0) or 0) % _FLASH_SWAP_PERIOD
        return (calls + offset) % _FLASH_SWAP_PERIOD == 0
    return calls % _FLASH_SWAP_PERIOD == 0


def _flash_swap_observe(layer, topk_ids) -> None:
    """入口:累计命中 + 周期触发重排(仅热/冷模式;CUDA graph 捕获期跳过)。"""
    if not _FLASH_TIER_SWAP_ON or not getattr(layer, "_flash_hotcold", False):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    state = getattr(layer, "_flash_swap", None)
    if state is None:
        return
    _flash_swap_bump(state, topk_ids, _FLASH_SWAP_DECAY)
    state["calls"] += 1
    if not _flash_swap_phase_due(state):
        return
    try:
        _flash_swap_step(layer)
    except Exception as _e:  # noqa: BLE001
        _FLASH_SWAP_STATS["errors"] += 1
        logger.warning_once("FLASH-SWAP step failed: %r", _e)


def _flash_swap_provision(layer, device) -> None:
    """加载期预建设备热度缓冲(供图外触发);必须在任何 CUDA graph 捕获之前完成。"""
    state = getattr(layer, "_flash_swap", None)
    if state is None or state.get("heat_gpu") is not None:
        return
    try:
        state["heat_gpu"] = torch.zeros(state["num_experts"], dtype=torch.int64,
                                     device=device)
    except Exception as _e:  # noqa: BLE001
        logger.warning_once("FLASH-SWAP: heat_gpu 预建失败 %r", _e)
    # 注:P2 预取暂存区**不在此分配** —— provision 调用点在加载早期(weights.py:271),
    # 那时 w13_tm_scales/_flash_cold_pinned 尚未建立 ⇒ 会静默跳过。改到 _flash_swap_step
    # 首调时惰性分配(那时槽必已就绪)。


def _flash_route_accum_guard(target, raw) -> None:
    """值域守卫:越界 expert id(-1 / >=E)零贡献(clamp + 权重掩码,形状静态 ⇒ 图安全)。

    M1(2026-10-10)单一真源:件② 换权热度(`_flash_swap_accum_device`)与路由统计
    (`telemetry._flash_routing_accum_device`)此前各写一遍同法,现共用本函数。
    干净数据下权重恒 1 ⇒ 与旧 `torch.ones_like(ids)` 逐位等价(零行为变化)。
    布尔索引会产生**数据相关形状**、破坏 CUDA graph 捕获,故用 clamp + 掩码而非布尔索引。
    """
    n = target.numel()
    weights = ((raw >= 0) & (raw < n)).to(torch.int64)
    target.index_add_(0, raw.clamp(0, n - 1), weights)


def _flash_swap_accum_device(layer, topk_ids) -> None:
    """图内:设备侧累积命中(可被 CUDA graph 捕获;不做任何 CPU 同步)。

    L5-13(审计·2026-10-03):越界 id(如 -1 / >=E)必须**零贡献**,与 eager 路
    `_flash_swap_bump` 的值域过滤对齐。这里用「clamp + 权重掩码」而非布尔索引:
    布尔索引会产生**数据相关形状**,破坏 CUDA graph 捕获;本写法形状全程静态 ⇒ 图安全。
    """
    state = getattr(layer, "_flash_swap", None)
    if state is None:
        return
    raw = topk_ids.reshape(-1)
    if raw.dtype != torch.int64:
        raw = raw.to(torch.int64)
    hg = state.get("heat_gpu")
    if hg is None:
        try:
            hg = torch.zeros(state["num_experts"], dtype=torch.int64, device=raw.device)
            state["heat_gpu"] = hg
        except Exception:  # noqa: BLE001   捕获期分配失败则本次不记(下次 eager 再建)
            return
    if hg.device != raw.device:
        logger.warning_once(
            "FLASH-SWAP: heat_gpu(%s) 与路由 id(%s) 设备不一致 -> 本层热度不累积(换权停摆)",
            hg.device, raw.device)
        return
    # 值域守卫:越界 id 零贡献。单一真源 M1(2026-10-10):下沉到
    # `_flash_route_accum_guard`,件② 热度与路由统计(telemetry)共用。
    _flash_route_accum_guard(hg, raw)


def _flash_swap_tick_graph() -> None:
    """图外:回读设备热度 -> 更新 CPU 热表 -> 周期重排(换权)。每步一次。"""
    if not _FLASH_TIER_SWAP_ON or not _FLASH_SWAP_GRAPH:
        return
    if not _FLASH_SWAP_TICK_SEEN[0]:
        _FLASH_SWAP_TICK_SEEN[0] = True
        logger.info("FLASH-SWAP: 图外 tick 首次生效(layer 数=%d)", len(_FLASH_SWAP_LAYERS))
        # R1 可观测:相位错开是否生效必须喊出来(静默=事故)。
        logger.info("FLASH-SWAP: R1 相位错开 stagger=%s (PERIOD=%d WARMUP=%d)",
                    _FLASH_SWAP_STAGGER, _FLASH_SWAP_PERIOD, _FLASH_SWAP_WARMUP)
    # ── 批量回读(性能):先把各层设备热度一次性 stack [L,E] 再**单次** D2H 同步 ──
    # 旧版逐层 `clone().to('cpu')` = 48 次 PCIe 往返/步(实测 ~0.66ms/步);
    # stack 后单次回读 ~0.05ms/步(容器内微基准 2026-09-30)。形状不一致时回退逐层。
    active = []
    for layer in list(_FLASH_SWAP_LAYERS):
        state = getattr(layer, "_flash_swap", None)
        if state is None:
            continue
        hg = state.get("heat_gpu")
        if hg is None:
            continue
        active.append((layer, state, hg))
    if not active:
        return
    deltas = None
    if len({tuple(h.shape) for _, _, h in active}) == 1:
        try:
            # ⚠ L4-F5:此处 `.to("cpu")` 为**阻塞**拷贝,兼作「排空当前 stream」的隐式屏障 ——
            #   紧随其后的 `_flash_swap_install` 依赖它确保上一步设备工作已完成。勿改 non_blocking。
            deltas = torch.stack([h.detach() for _, _, h in active]).to(
                "cpu", dtype=torch.int64)
            if not _FLASH_SWAP_BATCH_SEEN[0]:
                _FLASH_SWAP_BATCH_SEEN[0] = True
                logger.info("FLASH-SWAP: tick 批量回读生效(单次同步, 层数=%d)", len(active))
        except Exception as _e:  # noqa: BLE001
            logger.warning_once("FLASH-SWAP: tick 批量回读失败,回落逐层: %r", _e)
            deltas = None
    for i, (layer, state, hg) in enumerate(active):
        # P2:先把上一步预取的换权**安装**掉(必须在本步前向之前,保证布局一致)
        if state.get("pending"):
            try:
                _flash_swap_install(layer)
            except Exception as _e:  # noqa: BLE001
                logger.warning_once("FLASH-SWAP install failed: %r", _e)
        if deltas is not None:
            delta = deltas[i]
            hg.zero_()
        else:
            delta = hg.detach().clone().to("cpu", dtype=torch.int64)
            hg.zero_()
        if int(delta.sum()) == 0:
            # 空闲层也要推进步数计:否则该层热度的「距上次命中步数(age)」不涨、
            # 衰减停在旧值,唤醒后按过期热度判 t1 会换错对象。
            state["t"] += 1
            continue
        nz = torch.nonzero(delta, as_tuple=False).reshape(-1)
        _flash_swap_merge_counts(state, nz, delta[nz], _FLASH_SWAP_DECAY)
        state["calls"] += 1
        if not _flash_swap_phase_due(state):
            continue
        try:
            _flash_swap_step(layer)
        except Exception as _e:  # noqa: BLE001
            _FLASH_SWAP_STATS["errors"] += 1
            logger.warning_once("FLASH-SWAP step failed: %r", _e)


_FLASH_SWAP_TARGETS = (
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),  # V1
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),  # V2(Qwen4Exp 走这套)
)
_FLASH_SWAP_TICK_SEEN = [False]
_FLASH_SWAP_BATCH_SEEN = [False]


def _flash_swap_install_graph_hook() -> int:
    """把图外触发钩子挂到每个 GPUModelRunner 变体的 execute_model 头部(每步一次)。

    本引擎同时存在 V1(`vllm.v1.worker.gpu_model_runner`)与 V2
    (`vllm.v1.worker.gpu.model_runner`)两套 runner;Qwen4Exp 由 config 强制走 V2。
    两套都挂,确保真正被调用的那个一定被补上。
    """
    if not (_FLASH_TIER_SWAP_ON and _FLASH_SWAP_GRAPH):
        return 0
    import importlib

    installed = 0
    for modname, clsname in _FLASH_SWAP_TARGETS:
        try:
            cls = getattr(importlib.import_module(modname), clsname)
        except Exception as _e:  # noqa: BLE001
            logger.warning("FLASH-SWAP: 图外钩子导入失败 %s.%s %r", modname, clsname, _e)
            continue
        if getattr(cls, "_flash_swap_hooked", False):
            installed += 1
            continue
        _orig = cls.execute_model

        def _flash_execute_model(self, *args, _orig=_orig, **kwargs):
            try:
                _flash_swap_tick_graph()
            except Exception as _e:  # noqa: BLE001
                logger.warning_once("FLASH-SWAP tick failed: %r", _e)
            return _orig(self, *args, **kwargs)

        cls.execute_model = _flash_execute_model
        cls._flash_swap_hooked = True
        installed += 1
        logger.info("FLASH-SWAP: 图外触发钩子已挂载(%s.%s)", modname, clsname)
    if installed == 0:
        logger.warning("FLASH-SWAP: 图外触发钩子**一个都没挂上**(targets=%s)",
                       [f"{m}.{c}" for m, c in _FLASH_SWAP_TARGETS])
    return installed


def _flash_swap_report() -> None:
    if not _FLASH_TIER_SWAP_ON:
        return
    try:
        logger.info("FLASH-SWAP: 汇总(P=%d t1=%.2f decay=%.3f) %s",
                    _FLASH_SWAP_PERIOD, _FLASH_SWAP_T1,
                    _FLASH_SWAP_DECAY, _FLASH_SWAP_STATS)
    except Exception:  # noqa: BLE001
        pass


# =====================================================================================
