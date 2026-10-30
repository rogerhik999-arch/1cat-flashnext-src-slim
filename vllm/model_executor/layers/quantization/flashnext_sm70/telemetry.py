# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.telemetry — 观测:逐 token/逐层路由统计 + 在线命中统计 + 热集表/全序表装载

(ds-cloud 两档存储补丁;由 nvfp4_sm70_moe.py 机械抽出,行为不变)
"""
from __future__ import annotations

import os
import re
import torch
from vllm.logger import init_logger

logger = init_logger("vllm.model_executor.layers.quantization.nvfp4_sm70_moe")  # 同一 logger 名,保持日志前缀一致

from .tier_swap import (
    _FLASH_SWAP_GRAPH, _FLASH_SWAP_SLOT_NAMES, _FLASH_TIER_SWAP_ON,
    _flash_route_accum_guard, _flash_swap_accum_device, _flash_swap_observe,
)

_FLASH_ROUTING_ON = (
    (os.getenv("FLASH_ROUTING_LOG", "0") or "").strip().lower() not in ("", "0", "false", "no", "off")
    # 阶段2c:hitstat 打开时也必须采集计数(否则无数据可累计)
    or bool((os.getenv("FLASH_NVFP4_HITSTAT", "") or "").strip())
)
_FLASH_ROUTING_STATS: dict = {}
_FLASH_ROUTING_ORDER: list = []
_FLASH_ROUTING_CALLS: list = [0]
# N-05(2026-10-08):越界/负 expert id 被过滤的累计计数(仅在记录到非法 id 时报警一次)。
_FLASH_ROUTING_INVALID: list = [0]

# ── V2.3(2026-10-08):设备侧图内直方图 ─────────────────────────────────────────
# 问题:CUDA graph 重放不重跑 Python apply() ⇒ 原 CPU 直方图只记 eager 步(偏 prefill),
#   图内 decode 不被计入。修法:在**捕获期**对每层设备直方图做图安全 index_add_
#   (与 tier_swap 换权热度同法:clamp + 权重掩码,形状静态),被烘进图 ⇒ 随重放累积真实
#   decode 路由;图外按固定频率回读并入 CPU 统计。env FLASH_ROUTING_DEVICE_HIST 默认 1
#   (=0 回旧行为)。
_FLASH_ROUTING_DEV = (
    (os.getenv("FLASH_ROUTING_DEVICE_HIST", "1") or "1").strip().lower()
    not in ("", "0", "false", "no", "off")
)
# id(layer) -> {"layer","counts_gpu":int64[E],"last_read":int64[E],"abs_index":int}
_FLASH_ROUTING_DEV_LAYERS: dict = {}
_FLASH_ROUTING_DEV_MISS = [False]


def _flash_layer_index_from_name(layer):
    """从模块名解析绝对模型层号 `layers.N`;不可解析返回 -1。

    M1(2026-10-10)单一真源:`hotcold._flash_resolve_layer_index`(带调用序兜底)
    与 `_flash_routing_abs_index`(无副作用)此前各写一遍同一正则口径,现共用本函数。
    """
    for attr in ("layer_name", "prefix", "name"):
        value = getattr(layer, attr, None)
        if isinstance(value, str):
            matches = re.findall(r"layers\.(\d+)", value)
            if matches:
                return int(matches[-1])
    return -1


def _flash_routing_abs_index(layer, fallback):
    """从模块名解析**绝对模型层号 layers.N**(无副作用版)。

    与 `hotcold._flash_resolve_layer_index` 同口径(共用 `_flash_layer_index_from_name`),
    但**不做调用序兜底递增**(避免在观测路径上扰动 hotcold 的兜底计数);不可解析时返回 fallback。
    """
    index = _flash_layer_index_from_name(layer)
    return fallback if index < 0 else index


def _flash_routing_num_experts(layer) -> int:
    n = int(getattr(layer, "num_experts", 0) or 0)
    if n <= 0:
        n = int((getattr(layer, "_flash_swap", None) or {}).get("num_experts", 0) or 0)
    return n


def _flash_routing_provision(layer) -> None:
    """eager 期登记该层的设备直方图(必须在任何 CUDA graph 捕获**之前**完成)。

    惰性 + 幂等;分配失败(如无 CUDA / 无槽)→ 不登记,图内路自动跳过(退回旧行为)。
    """
    if not _FLASH_ROUTING_DEV:
        return
    key = id(layer)
    if key in _FLASH_ROUTING_DEV_LAYERS:
        return
    n = _flash_routing_num_experts(layer)
    if n <= 0:
        return
    try:
        dev = None
        for name in _FLASH_SWAP_SLOT_NAMES:
            tensor = getattr(layer, name, None)
            if tensor is not None:
                dev = tensor.device
                break
        if dev is None:
            return
        _FLASH_ROUTING_DEV_LAYERS[key] = {
            "layer": layer,
            "counts_gpu": torch.zeros(n, dtype=torch.int64, device=dev),
            "last_read": torch.zeros(n, dtype=torch.int64),
            # N-04(2026-10-08):图内 token 计数(标量,图安全 add_),与 counts_gpu 同步回读
            "tokens_gpu": torch.zeros((), dtype=torch.int64, device=dev),
            "tokens_read": 0,
            "abs_index": _flash_routing_abs_index(layer, -1),
        }
    except Exception as _e:  # noqa: BLE001
        logger.warning_once("FLASH-ROUTING-DEV: 设备直方图预建失败(该层退回 CPU 路): %r", _e)


def _flash_routing_accum_device(layer, topk_ids) -> None:
    """图内:对设备直方图做**图安全** index_add_(可被 CUDA graph 捕获;无 CPU 同步)。

    值域守卫与 `tier_swap._flash_swap_accum_device` 同法(clamp + 权重掩码,形状静态),
    越界 id(-1 / >=E)**零贡献**,不破坏 capture。
    """
    rec = _FLASH_ROUTING_DEV_LAYERS.get(id(layer))
    if rec is None:
        if not _FLASH_ROUTING_DEV_MISS[0]:
            _FLASH_ROUTING_DEV_MISS[0] = True
            logger.warning_once(
                "FLASH-ROUTING-DEV: 捕获期未见设备直方图(未在 eager 期预建?)-> 本次不记")
        return
    raw = topk_ids.reshape(-1)
    if raw.dtype != torch.int64:
        raw = raw.to(torch.int64)
    cg = rec["counts_gpu"]
    if cg.device != raw.device:
        logger.warning_once(
            "FLASH-ROUTING-DEV: counts_gpu(%s) 与路由 id(%s) 设备不一致 -> 本层不累积",
            cg.device, raw.device)
        return
    # 值域守卫:越界 id(-1 / >=E)零贡献(单一真源
    # tier_swap._flash_route_accum_guard,clamp + 掩码,形状静态 ⇒ 图安全)。
    _flash_route_accum_guard(cg, raw)
    # N-04:图内 token 计数(行数=token 数;形状静态常量,可被 CUDA graph 捕获)
    rec["tokens_gpu"].add_(int(topk_ids.shape[0]))


def _flash_routing_readback() -> None:
    """图外:把各层设备直方图的**增量**并入 CPU 统计(绝对层号;含图内 decode 命中)。

    以 last_read 记账 ⇒ 幂等、不双计。可在 dump/hitstat 前任意次数调用。
    """
    if not _FLASH_ROUTING_DEV_LAYERS:
        return
    for key, rec in list(_FLASH_ROUTING_DEV_LAYERS.items()):
        try:
            now = rec["counts_gpu"].detach().to("cpu", dtype=torch.int64).clone()
            _tg = rec.get("tokens_gpu")
            tok_now = int(_tg.item()) if _tg is not None else int(rec.get("tokens_read", 0))
        except Exception:  # noqa: BLE001
            continue
        delta = now - rec["last_read"]
        dsum = int(delta.sum())
        dtok = tok_now - int(rec.get("tokens_read", 0))
        if dsum <= 0 and dtok <= 0:   # N-04:tokens 变而 counts 求和为 0 的极端也并入
            continue
        rec["last_read"] = now
        rec["tokens_read"] = tok_now
        stats = _FLASH_ROUTING_STATS.get(key)
        if stats is None:
            idx = rec["abs_index"]
            stats = {
                "layer_index": idx if idx >= 0 else len(_FLASH_ROUTING_ORDER),
                "tokens": 0,
                # N-05:槽数 = 设备直方图长度(= 真实 E),与 CPU/设备路同口径
                "counts": torch.zeros(int(rec["counts_gpu"].numel()), dtype=torch.int64),
            }
            _FLASH_ROUTING_STATS[key] = stats
            _FLASH_ROUTING_ORDER.append(key)
        elif rec["abs_index"] >= 0:
            stats["layer_index"] = rec["abs_index"]
        take = min(delta.numel(), stats["counts"].numel())
        stats["counts"][:take] += delta[:take]
        stats["tokens"] += dtok   # ★N-04:并入图内 decode token(旧版只并入 counts)
        stats["graph_hits"] = int(stats.get("graph_hits", 0)) + dsum


# --- P0.1 逐 token 路由序列日志(opt-in;给「动态换专家」做时间局部性 go/no-go) ---
_FLASH_ROUTING_SEQ_ON = (
    (os.getenv("FLASH_ROUTING_SEQ_LOG", "0") or "").strip().lower()
    not in ("", "0", "false", "no", "off")
)
_FLASH_ROUTING_SEQ_STEP: list = [0]
_FLASH_ROUTING_SEQ_HANDLE: list = [None]
_FLASH_ROUTING_SEQ_LAYER_ORDINAL: dict = {}
_FLASH_ROUTING_SEQ_MAX_TOKENS = int(
    os.getenv("FLASH_ROUTING_SEQ_MAX_TOKENS", "32") or "32")


def _flash_routing_seq_layer_ordinal(layer) -> int:
    """该层在「本次运行」中的**登记序号**(首次见到即编号,稳定不变)。

    仅用于逐 token 路由序列日志(flash_routing_seq_*.tsv)的 `layer` 列。
    ⚠️ 与 `hotcold._flash_resolve_layer_index`(解析出的**模型层号**)是**两个不同概念**:
    本函数给的是「第几个被登记的层」,不是 layers.N 里的 N。
    """
    key = id(layer)
    value = _FLASH_ROUTING_SEQ_LAYER_ORDINAL.get(key)
    if value is None:
        value = len(_FLASH_ROUTING_SEQ_LAYER_ORDINAL)
        _FLASH_ROUTING_SEQ_LAYER_ORDINAL[key] = value
    return value


def _flash_record_routing_seq(layer, topk_ids) -> None:
    """按 MoE 调用顺序落盘 top-k 专家 id(仅 decode 段;一行一次调用)。"""
    # CUDA graph 捕获期间禁止 CPU 同步(会破坏 capture)
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    ids = topk_ids.detach().to("cpu", dtype=torch.int64)
    if ids.dim() < 2:
        ids = ids.reshape(1, -1)
    n_tokens = int(ids.shape[0])
    if n_tokens > _FLASH_ROUTING_SEQ_MAX_TOKENS:
        return  # 只记 decode(小批),prefill 不记
    handle = _FLASH_ROUTING_SEQ_HANDLE[0]
    if handle is None:
        path = (os.getenv("FLASH_ROUTING_SEQ_OUT", "") or "").strip() or os.path.join(
            os.getenv("FLASH_ROUTING_DIR", "/tmp"),
            "flash_routing_seq_%d.tsv" % os.getpid())
        handle = open(path, "w", encoding="utf-8", buffering=1)
        handle.write("call\tlayer\tn_tokens\texperts\n")
        _FLASH_ROUTING_SEQ_HANDLE[0] = handle
        logger.info("FLASH-ROUTING-SEQ: -> %s", path)
    step = _FLASH_ROUTING_SEQ_STEP[0]
    _FLASH_ROUTING_SEQ_STEP[0] = step + 1
    layer_ordinal = _flash_routing_seq_layer_ordinal(layer)
    flat = ",".join(str(int(x)) for x in ids.reshape(-1).tolist())
    handle.write("%d\t%d\t%d\t%s\n" % (step, layer_ordinal, n_tokens, flat))


def _flash_flush_routing_seq() -> None:
    handle = _FLASH_ROUTING_SEQ_HANDLE[0]
    if handle is not None:
        try:
            handle.flush()
            handle.close()
        except Exception as _e:  # noqa: BLE001
            logger.debug("FLASH-ROUTING: 序列日志 flush/close 失败(忽略): %r", _e)


def _flash_routing_accum_cpu(layer, topk_ids) -> None:
    """图外 CPU 直方图(与设备路互补);越界 id 不计入、负 id 不抛(Batch5 V2.4-2)。

    正常 id∈[0,512) 时结果与旧 `bincount(ids, minlength=4096)` **逐位相同**;
    越界/负 id 仅被过滤,不影响合法计数。异常由调用方隔离。
    """
    key = id(layer)
    stats = _FLASH_ROUTING_STATS.get(key)
    if stats is None:
        # N-05:槽数 = 真实专家数(与设备路同口径);拿不到才退回防御性上界 4096。
        slots = _flash_routing_num_experts(layer)
        if slots <= 0:
            slots = 4096
        stats = {
            # 绝对模型层号(V2.3,与 hotcold/热表装载 layers.N 对齐);不可解析时退回登记序
            "layer_index": _flash_routing_abs_index(layer, len(_FLASH_ROUTING_ORDER)),
            "tokens": 0,
            "counts": torch.zeros(slots, dtype=torch.int64),
        }
        _FLASH_ROUTING_STATS[key] = stats
        _FLASH_ROUTING_ORDER.append(key)
    n = stats["counts"].numel()
    raw = topk_ids.detach().to("cpu", dtype=torch.int64).reshape(-1)
    valid = raw[(raw >= 0) & (raw < n)]        # 过滤越界/负 id(不抛、不计入)
    invalid = int(raw.numel() - valid.numel())
    if invalid:
        _FLASH_ROUTING_INVALID[0] += invalid
        logger.warning_once(
            "FLASH-ROUTING: %d 个越界/负 expert id 被过滤(累计 %d;上界=%d)",
            invalid, _FLASH_ROUTING_INVALID[0], n)
    # valid 已限定 <n ⇒ bincount 结果长度必为 n(torch.bincount 无 maxlength 参数)。
    stats["counts"] += torch.bincount(valid, minlength=n)
    stats["tokens"] += int(topk_ids.shape[0])


def _flash_record_routing(layer, topk_ids) -> None:
    """把每次 MoE 调用的 topk_ids 累加到按层统计里(仅 FLASH_ROUTING_LOG 打开时)。"""
    if _FLASH_ROUTING_SEQ_ON:
        try:
            _flash_record_routing_seq(layer, topk_ids)
        except Exception as _e:  # noqa: BLE001
            logger.warning_once("FLASH-ROUTING-SEQ failed: %r", _e)
    if _FLASH_TIER_SWAP_ON:
        if _FLASH_SWAP_GRAPH:
            if not getattr(layer, "_flash_heat_via_router", False):
                try:
                    _flash_swap_accum_device(layer, topk_ids)
                except Exception as _e:  # noqa: BLE001
                    logger.warning_once("FLASH-SWAP accum failed: %r", _e)
        else:
            try:
                _flash_swap_observe(layer, topk_ids)
            except Exception as _e:  # noqa: BLE001
                logger.warning_once("FLASH-SWAP observe failed: %r", _e)
    if not _FLASH_ROUTING_ON:
        return
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if capturing:
        # 图内:图安全设备直方图(V2.3)—— 被烘进图 ⇒ 随重放累积真实 decode 路由;
        # 不在此做任何 CPU 同步(会破坏 capture)。
        if _FLASH_ROUTING_DEV:
            try:
                _flash_routing_accum_device(layer, topk_ids)
            except Exception as _e:  # noqa: BLE001
                logger.warning_once("FLASH-ROUTING-DEV accum failed: %r", _e)
        return
    # 图外(eager:涵盖 prefill 与未走图的 decode):CPU 直方图
    if _FLASH_ROUTING_DEV:
        _flash_routing_provision(layer)   # 预建设备直方图,供随后的图捕获使用
    try:
        _flash_routing_accum_cpu(layer, topk_ids)
    except Exception as _e:  # noqa: BLE001  (遥测异常必须与 forward 隔离,不得冒泡)
        logger.warning_once("FLASH-ROUTING: CPU 直方图累加失败(已隔离): %r", _e)
    _FLASH_ROUTING_CALLS[0] += 1
    if _FLASH_ROUTING_CALLS[0] % 50 == 0:
        _flash_dump_routing()


# ================= FLASH PATCH 阶段2c:在线累计命中统计 (ds-cloud 2026-09-26) ==============
# 目的(韩总 2026-09-26):「统计专家命中次数,超冷专家在加载时优先放硬盘」。
#   router 已按层累计计数,但进程退出时**覆盖**写 flash_routing_<pid>.tsv(一跑一清,
#   跨会话/跨 worker 无法累计)。本补丁加 env FLASH_NVFP4_HITSTAT=<json 路径>:
#   每次 dump 时把本进程**增量**并入该 JSON(读-改-写 + 文件锁 + 原子替换),
#   跨会话/跨 worker 持久累计。留空 = 与旧行为完全一致(零影响)。
#   输出 {"meta":{...},"per_layer":{"<layer>":{"<expert>":count}}},可直接喂
#   next/expert_stats.py --agg <该 json>。
_FLASH_HITSTAT_PATH = (os.getenv("FLASH_NVFP4_HITSTAT", "") or "").strip()
_FLASH_HITSTAT_ON = bool(_FLASH_HITSTAT_PATH)
_FLASH_HITSTAT_BASE: dict = {}  # layer_index -> 上次已并入的 counts(tensor),用于算增量
# 周期并入节流:_flash_dump_routing 每 50 次调用就跑一次(≈每 token 一次),若同频重写
# 大 JSON 开销不可忽略 → 周期路按「最小间隔秒」节流;退出时另走 force 路(不丢数据)。
_FLASH_HITSTAT_MIN_INTERVAL = float(os.getenv("FLASH_NVFP4_HITSTAT_MIN_INTERVAL", "30") or "30")
_FLASH_HITSTAT_LAST = [0.0]
# ── 异步落盘(2026-09-30 · P1):把「flock + JSON 读改写」移出引擎线程 ───────────────
# 原实现把 11.1ms 的 RMW(读大 JSON + 序列化 + 原子替换)压在**路由/前向路径**上,
# 仅靠 30s 节流摊薄(0.04%)。⇒ 统计一旦调频(调低 MIN_INTERVAL)即线性变贵。
# 现将「算增量」(便宜,引擎线程)与「落盘」(贵,后台线程)拆开:引擎线程只入队,
# 落盘耗时与频率**解耦**。默认开(=1);=0 回退同步旧行为。退出 flush 会等待队列排空。
_FLASH_HITSTAT_ASYNC = (os.getenv("FLASH_NVFP4_HITSTAT_ASYNC", "1") or "1").strip().lower() \
    not in ("", "0", "false", "no", "off")
_FLASH_HITSTAT_QUEUE = None
_FLASH_HITSTAT_WORKER = [None]
_FLASH_HITSTAT_WLOCK = None


def _flash_hitstat_write(deltas) -> None:
    """慢路径:flock 内 RMW + 原子替换(仅在后台线程 / 同步回退 / 退出时执行)。"""
    import fcntl
    import json as _json
    import time as _time

    path = _FLASH_HITSTAT_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path + ".lock", "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        except Exception as _e:  # noqa: BLE001
            # 无锁读-改-写在 TP2 双 worker 下会静默互盖丢计数 ⇒ 告警跳过本轮。
            logger.warning(
                "FLASH-HITSTAT: flock 获取失败(%r) -> 跳过本轮合并,下轮重试"
                "(避免无锁读改写丢计数)", _e)
            return
        doc = {"meta": {}, "per_layer": {}}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    loaded = _json.load(fh)
                if isinstance(loaded.get("per_layer"), dict):
                    doc = loaded
            except Exception:  # noqa: BLE001
                logger.warning("FLASH-HITSTAT: 现有文件不可读,重建 -> %s", path)
        per_layer = doc.setdefault("per_layer", {})
        for layer_index, diff in deltas.items():
            row = per_layer.setdefault(str(layer_index), {})
            for expert in torch.nonzero(diff, as_tuple=False).reshape(-1).tolist():
                key = str(expert)
                row[key] = int(row.get(key, 0)) + int(diff[expert])
        meta = doc.setdefault("meta", {})
        meta["updated_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
        meta["pid"] = os.getpid()
        meta["writers"] = sorted(set(meta.get("writers", [])) | {str(os.getpid())})
        tmp = "%s.tmp.%d" % (path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(doc, fh, ensure_ascii=False)
        os.replace(tmp, path)
    logger.info("FLASH-HITSTAT: 并入 %d 层 -> %s", len(deltas), path)


def _flash_hitstat_worker_loop() -> None:
    while True:
        item = _FLASH_HITSTAT_QUEUE.get()
        try:
            if item is None:
                return
            _flash_hitstat_write(item)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FLASH-HITSTAT worker failed: %r", exc)
        finally:
            try:
                _FLASH_HITSTAT_QUEUE.task_done()
            except Exception:  # noqa: BLE001
                pass


def _flash_hitstat_ensure_worker():
    """惰性起后台落盘线程(每进程一个,daemon)。"""
    global _FLASH_HITSTAT_QUEUE, _FLASH_HITSTAT_WLOCK
    import queue as _queue
    import threading as _threading

    if _FLASH_HITSTAT_WLOCK is None:
        _FLASH_HITSTAT_WLOCK = _threading.Lock()
    with _FLASH_HITSTAT_WLOCK:
        if _FLASH_HITSTAT_QUEUE is None:
            _FLASH_HITSTAT_QUEUE = _queue.Queue(maxsize=8)
        w = _FLASH_HITSTAT_WORKER[0]
        if w is None or not w.is_alive():
            t = _threading.Thread(
                target=_flash_hitstat_worker_loop, name="flash-hitstat", daemon=True)
            t.start()
            _FLASH_HITSTAT_WORKER[0] = t
    return _FLASH_HITSTAT_QUEUE


def _flash_hitstat_merge(stats_map) -> None:
    """把本进程路由计数**增量**并入 FLASH_NVFP4_HITSTAT(持久累计)。失败不阻断主流程。

    P1(2026-09-30):「算增量」留引擎线程(便宜);「落盘」异步交后台线程(贵)。
    BASE 在快照时推进 ⇒ 并发/重入不重复计数;极端写失败丢本轮增量(仅统计,可接受)。
    """
    if not _FLASH_HITSTAT_ON:
        return
    try:
        # 1) 算增量(自上次并入以来;clamp 防「层重建」时的负值)
        deltas = {}
        for _key, stats in stats_map.items():
            layer_index = int(stats["layer_index"])
            counts = stats["counts"]
            base = _FLASH_HITSTAT_BASE.get(layer_index)
            # ★克隆:首轮/形状不符时 diff 就是活动张量本体;若异步入队后引擎线程继续原地 +=,
            #   后台会读到被改写的值 -> 重复计数(审计 P2-3)。
            diff = (counts.clone() if (base is None or base.numel() != counts.numel())
                    else (counts - base).clamp(min=0))
            if int(diff.sum()) > 0:
                deltas[layer_index] = diff
            _FLASH_HITSTAT_BASE[layer_index] = counts.clone()
        if not deltas:
            return
        # 2) 落盘:异步入队(默认)或同步
        if _FLASH_HITSTAT_ASYNC:
            try:
                _flash_hitstat_ensure_worker().put_nowait(deltas)
            except Exception:  # 队列满/线程异常 -> 同步兜底(不丢)
                _flash_hitstat_write(deltas)
        else:
            _flash_hitstat_write(deltas)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FLASH-HITSTAT merge failed: %r", exc)


def _flash_hitstat_merge_now(throttle: bool) -> None:
    """并入当前路由直方图。throttle=True 走时间节流(周期 tick)；False 强制(退出 flush,不丢)。"""
    if not _FLASH_HITSTAT_ON:
        return
    if throttle:
        import time as _time
        now = _time.time()
        if now - _FLASH_HITSTAT_LAST[0] < _FLASH_HITSTAT_MIN_INTERVAL:
            return
        _FLASH_HITSTAT_LAST[0] = now
    _flash_hitstat_merge(_FLASH_ROUTING_STATS)


def _flash_hitstat_tick() -> None:
    """周期并入(带时间节流)。由 _flash_dump_routing 调用。"""
    _flash_hitstat_merge_now(throttle=True)


def _flash_hitstat_flush() -> None:
    """退出时强制并入(不节流,保证不丢);异步档需等后台队列排空后再返回。"""
    _flash_hitstat_merge_now(throttle=False)
    try:
        if _FLASH_HITSTAT_QUEUE is not None:
            _FLASH_HITSTAT_QUEUE.join()
    except Exception:  # noqa: BLE001
        pass


def _flash_dump_routing() -> None:
    """进程退出时把路由直方图落盘(每个 worker 一个文件,按 pid 区分)。"""
    if not _FLASH_ROUTING_ON:
        return
    _flash_routing_readback()   # V2.3:先并入图内设备直方图增量(含 decode)
    if not _FLASH_ROUTING_STATS:
        return
    out_path = os.getenv("FLASH_ROUTING_OUT") or os.path.join(
        os.getenv("FLASH_ROUTING_DIR", "/tmp"), f"flash_routing_{os.getpid()}.tsv"
    )
    try:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("layer_index	tokens	expert	sel_count\n")
            for _key, stats in sorted(
                _FLASH_ROUTING_STATS.items(), key=lambda item: item[1]["layer_index"]
            ):
                counts = stats["counts"]
                for expert in torch.nonzero(counts, as_tuple=False).reshape(-1).tolist():
                    handle.write(
                        f"{stats['layer_index']}\t{stats['tokens']}\t{expert}\t{int(counts[expert])}\n"
                    )
        logger.info("FLASH-ROUTING: dumped %d layers -> %s", len(_FLASH_ROUTING_STATS), out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FLASH-ROUTING dump failed: %r", exc)
    _flash_hitstat_tick()


# =====================================================================================

# ================= FLASH PATCH 阶段2b:逐层频次热集表 (ds-cloud 2026-09-25) ================
# 目的:按「每层真实路由频次」选热专家(而非按专家下标 list(range(h)))——实测两者覆盖
#       差距巨大(下标 0..223 仅 41.7%,逐层 top-224 按频次 ≈100%)。热集表由外部 JSON 提供。
_FLASH_LAYER_CALL = [0]
_FLASH_HOTLISTS = {"loaded": False, "map": None}


def _flash_load_per_layer_list(env_name, tag, transform):
    """载入「每层 -> 列表」JSON 表(`per_layer` 或裸映射);未配置返回 None。

    transform 决定元素归一化(热集=去重升序;全序表=保序)。热集表与全序表共用此加载器。
    """
    path = (os.getenv(env_name, "") or "").strip()
    if not path:
        return None
    try:
        import json
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        per_layer = data.get("per_layer", data)
        out = {int(k): transform(v) for k, v in per_layer.items()}
        logger.info("%s: 载入 %d 层 (%s)", tag, len(out), path)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s 载入失败: %r", tag, exc)
        return None


def _flash_load_hotlists():
    if _FLASH_HOTLISTS["loaded"]:
        return _FLASH_HOTLISTS["map"]
    _FLASH_HOTLISTS["loaded"] = True
    _FLASH_HOTLISTS["map"] = _flash_load_per_layer_list(
        "FLASH_NVFP4_HOT_LIST", "FLASH-HOTLIST",
        lambda v: sorted({int(x) for x in v}))
    return _FLASH_HOTLISTS["map"]


# ================= FLASH PATCH P0.2:专家全序表 + 两档预算 (ds-cloud 2026-09-26) ==============
# 目的:把「热专家 top-N」从**写死**改成**算出来**。
#   · FLASH_NVFP4_EXPERT_RANK   : 逐层专家全序表 JSON(由 next/make_expert_rank.py 生成)
#   · FLASH_NVFP4_GPU_EXPERT_BUDGET = off(默认) | auto | <N>   (KV 优先,显存专家额度)
# 全部 opt-in;不设 = 与旧行为完全一致。
_FLASH_EXPERT_RANK = {"loaded": False, "map": None}


def _flash_load_expert_rank():
    """载入逐层专家全序表(频次降序)。未配置返回 None。"""
    if _FLASH_EXPERT_RANK["loaded"]:
        return _FLASH_EXPERT_RANK["map"]
    _FLASH_EXPERT_RANK["loaded"] = True
    _FLASH_EXPERT_RANK["map"] = _flash_load_per_layer_list(
        "FLASH_NVFP4_EXPERT_RANK", "FLASH-RANK",
        lambda v: [int(x) for x in v])
    return _FLASH_EXPERT_RANK["map"]


