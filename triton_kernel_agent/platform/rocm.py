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

"""AMD ROCm (HIP / rocprofv3) implementations of platform interfaces.

ROCm PyTorch exposes HIP through the ``torch.cuda`` API, so verification,
benchmarking (``torch.cuda.Event`` -> HIP events) and the Triton launch
path are shared with the NVIDIA implementations.  What differs:

- **Profiling**: rocprofv3 hardware counters instead of NCU, turned into
  derived SOL / bandwidth / occupancy metrics.
- **GPU pinning**: ``HIP_VISIBLE_DEVICES`` overrides ``CUDA_VISIBLE_DEVICES``
  on ROCm, so workers set both.
- **Hardware description**: AMD specs (CU count, LDS, wavefront size)
  auto-detected from KFD sysfs when ``gpu_name`` is not given.
- **Wavefront size**: 64 on CDNA/Instinct, 32 on RDNA/gfx12.  It is
  detected per device and threaded into prompts and derived metrics.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from triton_kernel_agent.platform.amd_topology import AmdGpuNode, first_visible_gpu
from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    KernelProfilerBase,
)
from triton_kernel_agent.platform.nvidia import (
    NvidiaBenchmarker,
    NvidiaBottleneckAnalyzer,
    NvidiaRAGPrescriber,
    NvidiaRooflineAnalyzer,
    NvidiaVerifier,
    NvidiaWorkerRunner,
    _nvidia_worker_process,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manager-level components
# ---------------------------------------------------------------------------


class RocmVerifier(NvidiaVerifier):
    """Correctness verification; identical to NVIDIA since ROCm uses 'cuda'."""


class RocmBenchmarker(NvidiaBenchmarker):
    """Benchmarking via ``torch.cuda.Event``, which maps to HIP events."""


class RocmWorkerRunner(NvidiaWorkerRunner):
    """Spawns optimization workers pinned with HIP visibility env vars."""

    def _get_worker_target(self):
        return _rocm_worker_process


def _rocm_worker_process(*args: Any) -> None:
    """Pin to one GPU via ``HIP_VISIBLE_DEVICES``, then run the shared worker.

    When ``HIP_VISIBLE_DEVICES`` is set HIP ignores ``CUDA_VISIBLE_DEVICES``,
    so pinning only through the latter (as the NVIDIA runner does) silently
    fails if the user exported ``HIP_VISIBLE_DEVICES``.  The shared worker
    additionally sets ``CUDA_VISIBLE_DEVICES`` to the same id, which is
    harmless.
    """
    gpu_id = args[-1]
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    _nvidia_worker_process(*args)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

_MODEL_TOKEN = re.compile(r"(MI\d+[A-Z]?|RX\s*\d+\s*[A-Z]*|W\d+)", re.IGNORECASE)


def _amd_database_entries() -> dict[str, dict[str, Any]]:
    from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs_database import (
        GPU_SPECS_DATABASE,
    )

    return {
        name: dict(spec)
        for name, spec in GPU_SPECS_DATABASE.items()
        if spec.get("vendor") == "AMD"
    }


def _match_by_name(device_name: str) -> dict[str, Any] | None:
    entries = _amd_database_entries()
    if device_name in entries:
        return entries[device_name]
    # torch reports e.g. "AMD Instinct MI250X/MI250"; match on model token.
    wanted = {t.upper().replace(" ", "") for t in _MODEL_TOKEN.findall(device_name)}
    for name, spec in entries.items():
        tokens = {t.upper().replace(" ", "") for t in _MODEL_TOKEN.findall(name)}
        if tokens and tokens & wanted:
            return spec
    return None


def _match_by_topology(node: AmdGpuNode) -> dict[str, Any] | None:
    """Pick the database entry with the same gfx arch, CU count and memory size."""
    candidates = [
        spec
        for spec in _amd_database_entries().values()
        if spec.get("gfx_arch") == node.gfx_arch
        and spec.get("sm_count") == node.cu_count
    ]
    if node.memory_bytes and len(candidates) > 1:
        mem_gb = node.memory_bytes / 2**30
        candidates.sort(key=lambda s: abs(s.get("memory_gb", 0) - mem_gb))
    return candidates[0] if candidates else None


def specs_from_topology(node: AmdGpuNode) -> dict[str, Any]:
    """Best-effort specs for an AMD GPU missing from the database.

    Peak FLOPS / bandwidth are unknown, so the corresponding fields are
    omitted (the prompts skip missing fields; HBM % of peak is skipped).
    """
    specs: dict[str, Any] = {
        "name": f"AMD GPU ({node.gfx_arch})",
        "vendor": "AMD",
        "architecture": node.gfx_arch,
        "gfx_arch": node.gfx_arch,
        "wavefront_size": node.wavefront_size,
        "sm_count": node.cu_count,
        "max_threads_per_sm": node.max_threads_per_cu or None,
        "lds_kb": node.lds_kb or None,
        "memory_gb": round(node.memory_bytes / 2**30) if node.memory_bytes else None,
    }
    if node.l2_cache_kb:
        specs["l2_cache_mb"] = round(node.l2_cache_kb / 1024, 1)
    return {k: v for k, v in specs.items() if v is not None}


class RocmAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    """AMD GPU specs: database lookup by name, else by KFD topology.

    ``device_name=None`` auto-detects the first visible GPU without
    initializing HIP.
    """

    def __init__(self, topology_node: AmdGpuNode | None = None) -> None:
        self._node = topology_node

    def _topology(self) -> AmdGpuNode | None:
        if self._node is None:
            self._node = first_visible_gpu()
        return self._node

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        node = self._topology()
        specs = _match_by_name(device_name) if device_name else None
        if specs is None and node is not None:
            specs = _match_by_topology(node) or specs_from_topology(node)
        if specs is None:
            raise ValueError(
                f"Unknown AMD GPU {device_name!r} and no AMD GPU found in "
                "/sys/class/kfd; pass gpu_name matching a database entry"
            )
        # The live device is authoritative for wavefront size (e.g. when a
        # name match picked a sibling SKU).
        if node is not None:
            specs["wavefront_size"] = node.wavefront_size
        return specs


def _resolve_amd_specs(gpu_name: str | None) -> tuple[dict[str, Any], int]:
    """(specs, num_xcc) for the target GPU; degrades to ({}, 1) on failure."""
    provider = RocmAcceleratorSpecsProvider()
    try:
        specs = provider.get_specs(gpu_name)
    except ValueError as e:
        logger.warning(f"AMD GPU specs unavailable: {e}")
        specs = {}
    node = provider._topology()
    return specs, (node.num_xcc if node else 1)


def amd_hardware_notes(specs: dict[str, Any]) -> str:
    """Short AMD-specific context for the bottleneck-analysis prompt."""
    wave = specs.get("wavefront_size")
    wave_line = (
        f"- Wavefront size: {wave} lanes. Triton num_warps counts wavefronts, "
        f"so threads per program = num_warps * {wave}."
        if wave
        else "- Wavefront size: 32 or 64 depending on the GPU; "
        "threads per program = num_warps * wavefront size."
    )
    return f"""\
## AMD Hardware Notes
{wave_line}
- 'CU' is AMD's equivalent of an NVIDIA SM; LDS is the equivalent of shared memory.
- Matrix cores are MFMA (CDNA) or WMMA (RDNA/gfx12); 'Matrix Core Utilization' plays the role of tensor-core utilization.
- High VGPR/AGPR counts reduce resident wavefronts per SIMD (occupancy); nonzero scratch means register spills.
- 'Dependency Wait' is the share of wavefront cycles waiting on outstanding instructions (closest analogue to NCU scoreboard stalls).
- Recommend only fixes Triton supports on AMD (no TMA, warp specialization, or inline PTX); waves_per_eu and matrix_instr_nonkdim are AMD-specific knobs."""


# ---------------------------------------------------------------------------
# Worker-level components
# ---------------------------------------------------------------------------


class RocmKernelProfiler(KernelProfilerBase):
    """Wraps :class:`RocprofKernelProfiler` (rocprofv3) with lazy construction."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        rocprof_bin_path: str | None = None,
        rocprof_timeout_seconds: int | None = None,
        profiling_semaphore: Any | None = None,
        gpu_name: str | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else Path(".")
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._rocprof_bin_path = rocprof_bin_path
        self._rocprof_timeout_seconds = rocprof_timeout_seconds
        self._profiling_semaphore = profiling_semaphore
        self._gpu_name = gpu_name
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.profiling.rocprof_kernel_profiler import (
                RocprofKernelProfiler,
            )

            specs, num_xcc = _resolve_amd_specs(self._gpu_name)
            artifacts_dir = self._artifacts_dir or self._log_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "logger": self._logger,
                "artifacts_dir": artifacts_dir,
                "logs_dir": self._log_dir,
                "rocprof_bin_path": self._rocprof_bin_path,
                "profiling_semaphore": self._profiling_semaphore,
                "wavefront_size": specs.get("wavefront_size"),
                "peak_memory_bw_gbps": specs.get("peak_memory_bw_gbps"),
                "num_xcc": num_xcc,
            }
            if self._rocprof_timeout_seconds is not None:
                kwargs["rocprof_timeout_seconds"] = self._rocprof_timeout_seconds
            self._delegate = RocprofKernelProfiler(**kwargs)
        return self._delegate

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> Any | None:
        return self._get_delegate().profile_kernel(
            kernel_file, problem_file, round_num, max_retries
        )


class RocmRooflineAnalyzer(NvidiaRooflineAnalyzer):
    """Roofline analysis over rocprofv3-derived SOL metrics."""

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from kernel_perf_agent.kernel_opt.roofline.rocprof_roofline import (
                RocprofRooflineAnalyzer,
            )

            kwargs: dict[str, Any] = {"logger": self._logger}
            if self._roofline_config is not None:
                kwargs["config"] = self._roofline_config
            self._delegate = RocprofRooflineAnalyzer(**kwargs)
        return self._delegate


class RocmBottleneckAnalyzer(NvidiaBottleneckAnalyzer):
    """LLM bottleneck analysis fed with rocprofv3 metrics and AMD context.

    Unlike the NVIDIA version, ``gpu_name`` is optional (auto-detected).
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        openai_model: str = "gpt-5",
        gpu_name: str | None = None,
        roofline_config: Any | None = None,
        num_bottlenecks: int = 1,
    ) -> None:
        super().__init__(
            logger=logger,
            log_dir=log_dir,
            openai_model=openai_model,
            gpu_name=gpu_name,
            roofline_config=roofline_config,
            num_bottlenecks=num_bottlenecks,
        )
        self._roofline_config = roofline_config
        # Orchestrator accesses ``bottleneck_analyzer.roofline`` directly.
        self.roofline = RocmRooflineAnalyzer(
            logger=self._logger, roofline_config=roofline_config
        )

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from kernel_perf_agent.kernel_opt.diagnose_prompt.metric_schema import (
                ROCPROF_METRIC_SECTIONS,
            )
            from kernel_perf_agent.kernel_opt.roofline.rocprof_roofline import (
                RocprofRooflineAnalyzer,
            )
            from triton_kernel_agent.opt_worker_component.prescribing.bottleneck_analyzer import (
                BottleneckAnalyzer,
            )
            from utils.providers import get_model_provider

            specs, _ = _resolve_amd_specs(self._gpu_name)
            roofline_kwargs: dict[str, Any] = {"logger": self._logger}
            if self._roofline_config is not None:
                roofline_kwargs["config"] = self._roofline_config

            self._delegate = BottleneckAnalyzer(
                provider=get_model_provider(self._openai_model),
                model=self._openai_model,
                gpu_specs=specs,
                logs_dir=self._log_dir,
                logger=self._logger,
                num_bottlenecks=self._num_bottlenecks,
                roofline=RocprofRooflineAnalyzer(**roofline_kwargs),
                prompt_options={
                    "profiler_name": "rocprofv3",
                    "metric_sections": ROCPROF_METRIC_SECTIONS,
                    "matrix_unit_name": "Matrix Cores (MFMA/WMMA)",
                    "hardware_notes": amd_hardware_notes(specs),
                },
            )
        return self._delegate


class RocmRAGPrescriber(NvidiaRAGPrescriber):
    """RAG over the shared (Triton-level) optimization knowledge base.

    The knowledge base was written with NVIDIA in mind; AMD-specific
    entries are a follow-up.
    """
