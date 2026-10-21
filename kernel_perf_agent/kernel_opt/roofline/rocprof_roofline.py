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

"""
Roofline analysis for AMD GPUs using rocprofv3-derived SOL metrics.

AMD has no single "speed of light" counter like NCU's, so the profiler
derives them (see ``rocprof_profiler.derive_rocprof_metrics``):

- Compute SOL: max(VALUBusy, MfmaUtil)
- Memory SOL:  HBM bandwidth (FETCH_SIZE + WRITE_SIZE over kernel time) as %
  of the GPU's peak.  L2-bandwidth-bound kernels are not yet captured.

The classification logic is shared with the NCU analyzer.
"""

from kernel_perf_agent.kernel_opt.diagnose_prompt.metric_schema import (
    ROCPROF_COMPUTE_SOL_KEY,
    ROCPROF_MATRIX_UTIL_KEY,
    ROCPROF_MEMORY_SOL_KEY,
)
from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer


class RocprofRooflineAnalyzer(RooflineAnalyzer):
    """Roofline analyzer reading rocprofv3-derived SOL metrics."""

    COMPUTE_SOL_KEY = ROCPROF_COMPUTE_SOL_KEY
    MEMORY_SOL_KEY = ROCPROF_MEMORY_SOL_KEY
    TENSOR_UTIL_KEY = ROCPROF_MATRIX_UTIL_KEY
    PROFILER_NAME = "rocprofv3"
