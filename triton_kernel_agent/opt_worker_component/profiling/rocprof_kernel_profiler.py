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

"""Profiles Triton kernels on AMD GPUs using rocprofv3."""

import json
import logging
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from kernel_perf_agent.kernel_opt.profiler.rocprof_profiler import (
    find_rocprofv3,
    load_rocprof_counters,
    profile_triton_kernel_rocprof,
    rocprof_metrics_by_kernel,
    rocprof_version,
)
from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
    DEFAULT_NCU_TIMEOUT_SECONDS,
    KernelProfiler,
    ProfilerResults,
)

# Applied to *each* rocprofv3 counter pass (the workload runs once per pass).
DEFAULT_ROCPROF_TIMEOUT_SECONDS = DEFAULT_NCU_TIMEOUT_SECONDS


class RocprofKernelProfiler(KernelProfiler):
    """rocprofv3-based profiler producing NCU-shaped metric dicts.

    The returned ``ProfilerResults.metrics`` is ``{kernel_name: {metric: value}}``
    with the derived metric names from ``ROCPROF_METRIC_SECTIONS``.
    """

    PROFILER_LABEL = "rocprofv3"
    PROFILER_NAME = "rocprofv3"
    METRICS_FILE_SUFFIX = "rocprof_metrics"

    def __init__(
        self,
        logger: logging.Logger,
        artifacts_dir: Path,
        logs_dir: Path,
        rocprof_bin_path: str | None = None,
        rocprof_timeout_seconds: int = DEFAULT_ROCPROF_TIMEOUT_SECONDS,
        profiling_semaphore: Any | None = None,
        wavefront_size: int | None = None,
        peak_memory_bw_gbps: float | None = None,
        num_xcc: int = 1,
    ):
        """
        Args:
            logger: Logger instance
            artifacts_dir: Directory for profiling artifacts
            logs_dir: Directory for saving metric JSON files
            rocprof_bin_path: Path to rocprofv3 (auto-detect if None)
            rocprof_timeout_seconds: Timeout per counter pass in seconds
            profiling_semaphore: Semaphore serializing GPU profiling
            wavefront_size: 32 or 64; fallback when the profile does not
                report MAX_WAVE_SIZE
            peak_memory_bw_gbps: Peak HBM bandwidth, for bandwidth % of peak
            num_xcc: Number of XCDs (normalizes GRBM_GUI_ACTIVE)
        """
        super().__init__(
            logger=logger,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            ncu_bin_path=None,
            ncu_timeout_seconds=rocprof_timeout_seconds,
            profiling_semaphore=profiling_semaphore,
        )
        self.rocprof_bin_path = rocprof_bin_path
        self.wavefront_size = wavefront_size
        self.peak_memory_bw_gbps = peak_memory_bw_gbps
        self.num_xcc = num_xcc

    @cached_property
    def rocprof_version(self) -> str | None:
        try:
            return rocprof_version(find_rocprofv3(self.rocprof_bin_path))
        except FileNotFoundError:
            return None

    @property
    def profiler_version(self) -> str | None:
        return self.rocprof_version

    def _collect_once(
        self,
        wrapper_file: Path,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
    ) -> ProfilerResults:
        csv_paths = profile_triton_kernel_rocprof(
            benchmark_script=wrapper_file,
            workdir=self.artifacts_dir,
            out_subdir=f"rocprof_round_{round_num}",
            rocprof_bin=self.rocprof_bin_path,
            timeout=self.ncu_timeout_seconds,
        )
        counters_df = load_rocprof_counters(csv_paths)
        metrics = rocprof_metrics_by_kernel(
            counters_df,
            wavefront_size=self.wavefront_size,
            peak_memory_bw_gbps=self.peak_memory_bw_gbps,
            num_xcc=self.num_xcc,
        )
        if not metrics:
            # Mirror the NCU path, where an empty CSV surfaces as a parse error
            # and triggers a retry.
            raise json.JSONDecodeError("no kernels in rocprofv3 output", "", 0)

        metrics_df = pd.DataFrame(
            [{"Kernel Name": name, **vals} for name, vals in metrics.items()]
        )
        return self._make_results(
            metrics_df, metrics, kernel_file, problem_file, round_num
        )
