# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AMD GPU discovery via the KFD sysfs topology.

Reads ``/sys/class/kfd/kfd/topology/nodes/*/properties`` directly so that
GPU enumeration, wavefront size and basic hardware shape can be learned
**without initializing the HIP runtime**.  This matters for the same
reason ``_detect_gpus`` avoids ``torch.cuda`` in the manager: workers are
forked, and a parent that has already created a HIP context pins the
device list the children see.

Wavefront size is the main thing callers need.  AMD GPUs come in two
flavours:

- **wave64** — CDNA / Instinct (gfx908, gfx90a, gfx942, gfx950, ...)
- **wave32** — RDNA / Radeon (gfx10xx, gfx11xx, gfx12xx) and gfx1250

Triton's ``num_warps`` counts *wavefronts*, so the thread count of a
program is ``num_warps * wavefront_size``; getting this wrong silently
halves or doubles block sizes, which is why it is surfaced explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

KFD_TOPOLOGY_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")

# Explicit override, e.g. for generating kernels for a target that is not
# the local GPU.  Must be 32 or 64.
WAVEFRONT_SIZE_ENV = "KERNELAGENT_WAVEFRONT_SIZE"

VALID_WAVEFRONT_SIZES = (32, 64)


@dataclass(frozen=True)
class AmdGpuNode:
    """One GPU node as reported by KFD."""

    node_id: int
    gfx_arch: str  # e.g. "gfx942"
    wavefront_size: int
    cu_count: int
    simd_per_cu: int
    max_waves_per_simd: int
    lds_kb: int
    num_xcc: int
    max_engine_clk_mhz: int
    memory_bytes: int
    l2_cache_kb: int | None

    @property
    def max_threads_per_cu(self) -> int:
        return self.max_waves_per_simd * self.simd_per_cu * self.wavefront_size


def gfx_arch_from_target_version(version: int) -> str:
    """Convert KFD ``gfx_target_version`` to an LLVM target name.

    The encoding is ``major * 10000 + minor * 100 + stepping``, and the
    minor/stepping digits are printed in hex (``90010`` -> ``gfx90a``).
    """
    major = version // 10000
    minor = (version // 100) % 100
    stepping = version % 100
    return f"gfx{major}{minor:x}{stepping:x}"


def default_wavefront_size_for_arch(gfx_arch: str) -> int:
    """Wavefront size used by Triton for *gfx_arch* when nothing better is known.

    gfx9 (GCN / CDNA) is wave64; gfx10+ (RDNA, gfx12 incl. gfx1250) is wave32.
    """
    digits = gfx_arch.removeprefix("gfx")
    if digits.startswith(("8", "9")):
        return 64
    return 32


def _read_properties(path: Path) -> dict[str, int]:
    props: dict[str, int] = {}
    try:
        text = path.read_text()
    except OSError:
        return props
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                props[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return props


def _read_l2_kb(node_dir: Path) -> int | None:
    caches = node_dir / "caches"
    if not caches.is_dir():
        return None
    for cache in sorted(caches.iterdir()):
        props = _read_properties(cache / "properties")
        if props.get("level") == 2 and props.get("size"):
            return props["size"]
    return None


def _read_memory_bytes(node_dir: Path) -> int:
    banks = node_dir / "mem_banks"
    total = 0
    if banks.is_dir():
        for bank in banks.iterdir():
            total += _read_properties(bank / "properties").get("size_in_bytes", 0)
    return total


def list_gpu_nodes(root: Path | None = None) -> list[AmdGpuNode]:
    """Return GPU nodes in KFD order (the order HIP enumerates devices).

    CPU nodes (``simd_count == 0``) are skipped.  Returns ``[]`` on hosts
    without the amdgpu driver.
    """
    root = KFD_TOPOLOGY_ROOT if root is None else root
    if not root.is_dir():
        return []

    def _node_key(p: Path) -> int:
        try:
            return int(p.name)
        except ValueError:
            return 1 << 30

    nodes: list[AmdGpuNode] = []
    for node_dir in sorted(root.iterdir(), key=_node_key):
        props = _read_properties(node_dir / "properties")
        simd_count = props.get("simd_count", 0)
        target = props.get("gfx_target_version", 0)
        if simd_count <= 0 or target <= 0:
            continue
        gfx_arch = gfx_arch_from_target_version(target)
        simd_per_cu = props.get("simd_per_cu") or 4
        wave = props.get("wave_front_size") or default_wavefront_size_for_arch(gfx_arch)
        nodes.append(
            AmdGpuNode(
                node_id=_node_key(node_dir),
                gfx_arch=gfx_arch,
                wavefront_size=wave,
                cu_count=simd_count // simd_per_cu,
                simd_per_cu=simd_per_cu,
                max_waves_per_simd=props.get("max_waves_per_simd", 0),
                lds_kb=props.get("lds_size_in_kb", 0),
                num_xcc=props.get("num_xcc") or 1,
                max_engine_clk_mhz=props.get("max_engine_clk_fcompute", 0),
                memory_bytes=_read_memory_bytes(node_dir),
                l2_cache_kb=_read_l2_kb(node_dir),
            )
        )
    return nodes


def visible_device_ids_from_env() -> list[int] | None:
    """Parse the HIP device-visibility env vars, if any are set.

    HIP honours ``HIP_VISIBLE_DEVICES`` and, when that is unset,
    ``CUDA_VISIBLE_DEVICES``.  When ``HIP_VISIBLE_DEVICES`` is set,
    ``CUDA_VISIBLE_DEVICES`` is ignored.
    """
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            continue
        if ids:
            return ids
    return None


def first_visible_gpu(root: Path | None = None) -> AmdGpuNode | None:
    """The GPU that ``torch.cuda`` device 0 would map to, if detectable."""
    nodes = list_gpu_nodes(root)
    if not nodes:
        return None
    visible = visible_device_ids_from_env()
    if visible and 0 <= visible[0] < len(nodes):
        return nodes[visible[0]]
    return nodes[0]


def detect_wavefront_size(root: Path | None = None) -> int | None:
    """Wavefront size of the target GPU, or ``None`` if unknown.

    Resolution order: ``KERNELAGENT_WAVEFRONT_SIZE`` env, then KFD sysfs.
    """
    override = os.environ.get(WAVEFRONT_SIZE_ENV)
    if override:
        try:
            value = int(override)
        except ValueError:
            value = 0
        if value not in VALID_WAVEFRONT_SIZES:
            raise ValueError(
                f"{WAVEFRONT_SIZE_ENV}={override!r} is invalid; "
                f"expected one of {VALID_WAVEFRONT_SIZES}"
            )
        return value
    node = first_visible_gpu(root)
    return node.wavefront_size if node else None
