# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext_sm70.pinned_host — 精确字节 pinned host 分配器(绕开 torch 的 2 幂取整)

问题
----
`torch.empty(..., pin_memory=True)` 走 torch 的 pinned host 分配器,会把请求尺寸
round 到 **2 的幂**(同镜像实测:100→128、200→256、250→256、300→512 MiB)。冷专家
槽每层 4 张张量各被 round 一次 ⇒ 宿主冷档恒为 48×[256+128+64+32]MiB ≈ 46 GiB,
**与冷专家数/上下文无关**(实测 256K→8K 四档一字节不动)。⇒ 缩上下文、腾热集都省不
下宿主内存,白白被取整吃掉。

做法
----
绕过 torch 的 pinned 分配器,直接用 CUDA 运行时 `cudaHostAlloc(cudaHostAllocMapped)`
按**精确字节**分配,再用 `torch.from_numpy` 包成 CPU tensor。同镜像实测:
  ① `is_pinned()` 仍为 True —— torch 用运行时查询判定,不看自身分配器记录;
  ② `get_cuda_view_from_cpu_tensor` 走「已 pinned」分支 → `cudaHostGetDevicePointer`,
     UVA 设备指针 == 宿主指针(零拷贝);
  ③ 宿主写、GPU 读回一致。
⇒ 精确分配 + UVA 全保。

开关
----
`FLASH_NVFP4_EXACT_HOST_ALLOC=1`(默认 `0` = 原 torch.empty 行为,字节不变)。
"""
from __future__ import annotations

import ctypes
import os

import torch

_ON = (os.getenv("FLASH_NVFP4_EXACT_HOST_ALLOC", "0") or "0").strip().lower() not in (
    "", "0", "false", "no", "off")

# cudaHostAllocFlags: Default=0, Portable=1, Mapped=2, WriteCombined=4
_HOST_ALLOC_MAPPED = 2

_runtime = None
# 保活:内存由 cudaHostAlloc 持有(进程内不释放),包装对象须防 GC。
_keepalive: list = []


def enabled() -> bool:
    """精确分配是否开启(env 闸,默认关)。"""
    return _ON


def _cudart():
    global _runtime
    if _runtime is None:
        last = None
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
            try:
                lib = ctypes.CDLL(name)
                lib.cudaHostAlloc.restype = ctypes.c_int
                lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_size_t, ctypes.c_uint]
                _runtime = lib
                break
            except OSError as exc:  # noqa: PERF203
                last = exc
        if _runtime is None:
            raise RuntimeError("pinned_host: 无法加载 libcudart: %s" % last)
    return _runtime


def alloc_pinned_exact(nbytes: int) -> torch.Tensor:
    """精确 `nbytes` 的 pinned host 内存 → 1-D uint8 tensor(共享该内存、UVA 可见)。"""
    nbytes = int(nbytes)
    if nbytes <= 0:
        return torch.empty(0, dtype=torch.uint8)
    ptr = ctypes.c_void_p()
    rc = _cudart().cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes),
                                 ctypes.c_uint(_HOST_ALLOC_MAPPED))
    if rc != 0 or not ptr.value:
        raise RuntimeError("pinned_host: cudaHostAlloc(%d) 失败 rc=%d" % (nbytes, rc))
    import numpy as np
    carr = (ctypes.c_uint8 * nbytes).from_address(ptr.value)
    arr = np.ctypeslib.as_array(carr)
    t8 = torch.from_numpy(arr)
    _keepalive.append((carr, arr, t8))
    return t8


def alloc_pinned_tensor(count: int, shape, dtype) -> torch.Tensor:
    """(`count`,) + `shape` / `dtype` 的精确 pinned CPU tensor。"""
    numel = int(count)
    for d in shape:
        numel *= int(d)
    itemsize = torch.empty(0, dtype=dtype).element_size()
    t8 = alloc_pinned_exact(numel * itemsize)
    return t8.view(dtype).view((int(count),) + tuple(int(d) for d in shape))
