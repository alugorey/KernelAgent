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

"""CPU-only tests for the AMD ROCm platform (no GPU or rocprofv3 needed)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
    build_bottleneck_prompt,
)
from kernel_perf_agent.kernel_opt.diagnose_prompt.metric_schema import (
    ROCPROF_METRIC_SECTIONS,
)
from kernel_perf_agent.kernel_opt.profiler.rocprof_profiler import (
    derive_rocprof_metrics,
    filter_passes,
    load_rocprof_counters,
    parse_counter_listing,
    rocprof_metrics_by_kernel,
)
from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineResult
from kernel_perf_agent.kernel_opt.roofline.rocprof_roofline import (
    RocprofRooflineAnalyzer,
)
from triton_kernel_agent.opt_worker_component.searching.ptx_fingerprint import (
    normalize_amdgcn,
    ptx_hash_from_cache,
)
from triton_kernel_agent.platform import amd_topology
from triton_kernel_agent.platform.amd_topology import (
    AmdGpuNode,
    default_wavefront_size_for_arch,
    detect_wavefront_size,
    gfx_arch_from_target_version,
    list_gpu_nodes,
)
from triton_kernel_agent.platform.registry import registry
from triton_kernel_agent.platform.rocm import (
    RocmAcceleratorSpecsProvider,
    RocmWorkerRunner,
    _rocm_worker_process,
    amd_hardware_notes,
)
from triton_kernel_agent.platform_config import (
    get_platform,
    get_platform_choices,
    make_rocm_platform,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_node(root: Path, node_id: int, props: dict, l2_kb=None, mem_bytes=0):
    node = root / str(node_id)
    node.mkdir(parents=True)
    (node / "properties").write_text(
        "\n".join(f"{k} {v}" for k, v in props.items()) + "\n"
    )
    if l2_kb:
        cache = node / "caches" / "0"
        cache.mkdir(parents=True)
        (cache / "properties").write_text(f"level 2\nsize {l2_kb}\nsibling_map 1,0\n")
    if mem_bytes:
        bank = node / "mem_banks" / "0"
        bank.mkdir(parents=True)
        (bank / "properties").write_text(f"heap_type 1\nsize_in_bytes {mem_bytes}\n")


@pytest.fixture
def fake_kfd(tmp_path, monkeypatch):
    """KFD tree: CPU node, MI300X-like wave64 GPU, RDNA3-like wave32 GPU."""
    root = tmp_path / "nodes"
    _write_node(root, 0, {"simd_count": 0, "gfx_target_version": 0})
    _write_node(
        root,
        1,
        {
            "simd_count": 1216,
            "gfx_target_version": 90402,
            "wave_front_size": 64,
            "simd_per_cu": 4,
            "max_waves_per_simd": 8,
            "lds_size_in_kb": 64,
            "num_xcc": 8,
            "max_engine_clk_fcompute": 2100,
        },
        l2_kb=4096,
        mem_bytes=192 * 2**30,
    )
    _write_node(
        root,
        2,
        {
            "simd_count": 192,
            "gfx_target_version": 110000,
            "wave_front_size": 32,
            "simd_per_cu": 2,
            "max_waves_per_simd": 16,
            "lds_size_in_kb": 64,
        },
        mem_bytes=24 * 2**30,
    )
    for var in (
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "KERNELAGENT_WAVEFRONT_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(amd_topology, "KFD_TOPOLOGY_ROOT", root)
    return root


def _node(**overrides) -> AmdGpuNode:
    base = {
        "node_id": 1,
        "gfx_arch": "gfx942",
        "wavefront_size": 64,
        "cu_count": 304,
        "simd_per_cu": 4,
        "max_waves_per_simd": 8,
        "lds_kb": 64,
        "num_xcc": 8,
        "max_engine_clk_mhz": 2100,
        "memory_bytes": 192 * 2**30,
        "l2_cache_kb": 4096,
    }
    base.update(overrides)
    return AmdGpuNode(**base)


# ---------------------------------------------------------------------------
# Topology / wavefront detection
# ---------------------------------------------------------------------------


class TestAmdTopology:
    @pytest.mark.parametrize(
        "version,arch",
        [
            (90402, "gfx942"),
            (90010, "gfx90a"),
            (90500, "gfx950"),
            (110000, "gfx1100"),
            (120500, "gfx1250"),
        ],
    )
    def test_gfx_arch_from_target_version(self, version, arch):
        assert gfx_arch_from_target_version(version) == arch

    @pytest.mark.parametrize(
        "arch,wave",
        [
            ("gfx90a", 64),
            ("gfx942", 64),
            ("gfx950", 64),
            ("gfx1100", 32),
            ("gfx1201", 32),
            ("gfx1250", 32),
        ],
    )
    def test_default_wavefront_size(self, arch, wave):
        assert default_wavefront_size_for_arch(arch) == wave

    def test_list_gpu_nodes_skips_cpu(self, fake_kfd):
        nodes = list_gpu_nodes(fake_kfd)
        assert [n.gfx_arch for n in nodes] == ["gfx942", "gfx1100"]
        mi300, rdna = nodes
        assert (mi300.cu_count, mi300.wavefront_size, mi300.num_xcc) == (304, 64, 8)
        assert mi300.l2_cache_kb == 4096
        assert mi300.max_threads_per_cu == 2048
        assert (rdna.cu_count, rdna.wavefront_size) == (96, 32)
        assert rdna.max_threads_per_cu == 1024

    def test_detect_wavefront_follows_visible_device(self, fake_kfd, monkeypatch):
        assert detect_wavefront_size(fake_kfd) == 64
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
        assert detect_wavefront_size(fake_kfd) == 32

    def test_hip_visible_devices_wins_over_cuda(self, fake_kfd, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
        assert detect_wavefront_size(fake_kfd) == 32

    def test_env_override(self, fake_kfd, monkeypatch):
        monkeypatch.setenv("KERNELAGENT_WAVEFRONT_SIZE", "32")
        assert detect_wavefront_size(fake_kfd) == 32

    def test_env_override_invalid(self, fake_kfd, monkeypatch):
        monkeypatch.setenv("KERNELAGENT_WAVEFRONT_SIZE", "48")
        with pytest.raises(ValueError):
            detect_wavefront_size(fake_kfd)

    def test_no_driver(self, tmp_path):
        assert list_gpu_nodes(tmp_path / "missing") == []


# ---------------------------------------------------------------------------
# PlatformConfig
# ---------------------------------------------------------------------------


class TestRocmPlatformConfig:
    def test_registered(self):
        assert "rocm" in get_platform_choices()
        cfg = get_platform("rocm")
        assert cfg.name == "rocm"
        assert cfg.device_string == "cuda"
        assert cfg.cuda_hacks_to_strip == ()

    def test_wave64_guidance(self):
        cfg = make_rocm_platform(64)
        assert "wave64" in cfg.kernel_guidance
        assert "num_warps * 64" in cfg.kernel_guidance
        assert "Wavefront size is 64" in cfg.guidance_block

    def test_wave32_guidance(self):
        cfg = make_rocm_platform(32)
        assert "wave32" in cfg.kernel_guidance
        assert "num_warps * 32" in cfg.kernel_guidance
        assert "WMMA" in cfg.kernel_guidance
        assert "Wavefront size is 32" in cfg.guidance_block

    def test_unknown_wavefront_guidance_covers_both(self):
        cfg = make_rocm_platform(None)
        assert "warp_size" in cfg.kernel_guidance
        assert "32 or 64" in cfg.guidance_block


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


class TestRocmSpecs:
    def test_exact_name(self):
        specs = RocmAcceleratorSpecsProvider(_node()).get_specs("AMD Instinct MI300X")
        assert specs["peak_memory_bw_gbps"] == 5300
        assert specs["vendor"] == "AMD"

    def test_fuzzy_name(self):
        specs = RocmAcceleratorSpecsProvider(
            _node(gfx_arch="gfx90a", cu_count=110)
        ).get_specs("AMD Instinct MI250X/MI250")
        assert specs["name"] == "AMD Instinct MI250X"

    def test_topology_disambiguates_by_memory(self):
        mi300x = RocmAcceleratorSpecsProvider(_node()).get_specs(None)
        mi325x = RocmAcceleratorSpecsProvider(
            _node(memory_bytes=256 * 2**30)
        ).get_specs(None)
        assert mi300x["name"] == "AMD Instinct MI300X"
        assert mi325x["name"] == "AMD Instinct MI325X"

    def test_unknown_gpu_synthesized_from_topology(self):
        node = _node(gfx_arch="gfx1250", wavefront_size=32, cu_count=128)
        specs = RocmAcceleratorSpecsProvider(node).get_specs(None)
        assert specs["architecture"] == "gfx1250"
        assert specs["wavefront_size"] == 32
        assert specs["sm_count"] == 128
        assert "peak_memory_bw_gbps" not in specs

    def test_live_wavefront_overrides_database(self):
        # A name match on a wave64 entry still reports the device's wave size.
        node = _node(wavefront_size=32)
        specs = RocmAcceleratorSpecsProvider(node).get_specs("AMD Instinct MI300X")
        assert specs["wavefront_size"] == 32


# ---------------------------------------------------------------------------
# rocprofv3 parsing and derived metrics
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    "Correlation_Id,Dispatch_Id,Agent_Id,Queue_Id,Process_Id,Thread_Id,Grid_Size,"
    "Kernel_Id,Kernel_Name,Workgroup_Size,LDS_Block_Size,Scratch_Size,VGPR_Count,"
    "Accum_VGPR_Count,SGPR_Count,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp"
)


def _csv(path: Path, rows: list[tuple]) -> Path:
    """rows: (dispatch_id, kernel_name, counter, value, start_ns, end_ns, workgroup)."""
    lines = [_CSV_HEADER]
    for dispatch, name, counter, value, start, end, wg in rows:
        lines.append(
            f'{dispatch},{dispatch},"Agent 2",1,1,1,1048576,1,"{name}",{wg},4096,0,'
            f'32,8,48,"{counter}",{value},{start},{end}'
        )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def two_pass_csvs(tmp_path):
    torch_k = "void at::native::vectorized_elementwise_kernel<4>(int)"
    p0 = _csv(
        tmp_path / "p0.csv",
        [
            (1, torch_k, "FETCH_SIZE", 10.0, 0, 1000, 256),
            # warmup dispatch of the Triton kernel (must be ignored)
            (2, "add_kernel", "FETCH_SIZE", 1.0, 0, 9_000_000, 256),
            (2, "add_kernel", "SQ_WAVES", 1.0, 0, 9_000_000, 256),
            # last dispatch: 1 GiB read in 1 ms -> ~1073.7 GB/s
            (3, "add_kernel", "FETCH_SIZE", 1048576.0, 0, 1_000_000, 256),
            (3, "add_kernel", "SQ_WAVES", 4096.0, 0, 1_000_000, 256),
            (3, "add_kernel", "VALUBusy", 12.5, 0, 1_000_000, 256),
            (3, "add_kernel", "MfmaUtil", 0.0, 0, 1_000_000, 256),
        ],
    )
    p1 = _csv(
        tmp_path / "p1.csv",
        [
            (3, "add_kernel", "WRITE_SIZE", 524288.0, 0, 2_000_000, 256),
            (3, "add_kernel", "TCC_HIT_sum", 300.0, 0, 2_000_000, 256),
            (3, "add_kernel", "TCC_MISS_sum", 100.0, 0, 2_000_000, 256),
            (3, "add_kernel", "TCC_BUSY_avr", 400.0, 0, 2_000_000, 256),
            (3, "add_kernel", "GRBM_GUI_ACTIVE", 8000.0, 0, 2_000_000, 256),
        ],
    )
    return [p0, p1]


class TestRocprofParsing:
    def test_load_keeps_last_dispatch(self, two_pass_csvs):
        df = load_rocprof_counters(two_pass_csvs)
        add_rows = df[df["Kernel Name"] == "add_kernel"]
        assert set(add_rows["dispatch_id"]) == {3}
        assert len(add_rows) == 2  # one row per pass

    def test_derived_metrics_wave64(self, two_pass_csvs):
        df = load_rocprof_counters(two_pass_csvs)
        metrics = rocprof_metrics_by_kernel(
            df, wavefront_size=64, peak_memory_bw_gbps=5300, num_xcc=8
        )
        assert list(metrics) == ["add_kernel"]  # PyTorch kernel filtered
        m = metrics["add_kernel"]
        read_bw = 1048576 * 1024 / 1_000_000
        write_bw = 524288 * 1024 / 2_000_000
        assert m["hbm_bandwidth_gbps"] == pytest.approx(read_bw + write_bw, rel=1e-3)
        assert m["hbm_bw_pct_of_peak"] == pytest.approx(
            100 * (read_bw + write_bw) / 5300, rel=1e-3
        )
        assert m["l2_hit_rate_pct"] == pytest.approx(75.0)
        # GRBM_GUI_ACTIVE is summed over 8 XCDs -> 1000 cycles per XCD.
        assert m["l2_busy_pct"] == pytest.approx(40.0)
        assert m["compute_sol_pct"] == pytest.approx(12.5)
        assert m["memory_sol_pct"] == pytest.approx(m["hbm_bw_pct_of_peak"])
        assert m["waves_per_workgroup"] == 4
        assert m["wavefront_size"] == 64
        assert m["vgpr_count"] == 32 and m["agpr_count"] == 8

    def test_derived_metrics_wave32(self, two_pass_csvs):
        df = load_rocprof_counters(two_pass_csvs)
        m = rocprof_metrics_by_kernel(df, wavefront_size=32)["add_kernel"]
        assert m["waves_per_workgroup"] == 8
        assert "hbm_bw_pct_of_peak" not in m  # no peak given

    def test_profiled_max_wave_size_wins(self):
        rows = [{"MAX_WAVE_SIZE": 32.0, "Workgroup_Size": 256, "duration_ns": 1.0}]
        m = derive_rocprof_metrics(rows, wavefront_size=64)
        assert m["wavefront_size"] == 32
        assert m["waves_per_workgroup"] == 8

    def test_only_pytorch_kernels_kept_as_fallback(self, tmp_path):
        torch_k = "void at::native::reduce_kernel<512, 1>(int)"
        p = _csv(tmp_path / "p.csv", [(1, torch_k, "VALUBusy", 5.0, 0, 10, 256)])
        metrics = rocprof_metrics_by_kernel(load_rocprof_counters([p]))
        assert list(metrics) == [torch_k]

    def test_parse_counter_listing_first_gpu_only(self):
        text = (
            "GPU:0\nCounter_Name        :\tSQ_WAVES\n"
            "Counter_Name        :\tFETCH_SIZE\n"
            "GPU:1\nCounter_Name        :\tONLY_ON_GPU1\n"
        )
        assert parse_counter_listing(text) == {"SQ_WAVES", "FETCH_SIZE"}

    def test_filter_passes(self):
        passes = [["A", "B"], ["C"]]
        assert filter_passes(passes, frozenset({"A"})) == [["A"]]
        assert filter_passes(passes, frozenset()) == passes


class TestRocprofRoofline:
    def test_memory_bound(self):
        r = RocprofRooflineAnalyzer().analyze(
            {"compute_sol_pct": 10.0, "memory_sol_pct": 85.0, "mfma_util_pct": 0.0}
        )
        assert r.bottleneck == "memory"
        assert not r.uses_tensor_cores

    def test_matrix_cores_detected(self):
        r = RocprofRooflineAnalyzer().analyze(
            {"compute_sol_pct": 90.0, "memory_sol_pct": 20.0, "mfma_util_pct": 70.0}
        )
        assert r.bottleneck == "compute"
        assert r.uses_tensor_cores

    def test_missing_metrics_names_profiler(self):
        r = RocprofRooflineAnalyzer().analyze({})
        assert r.bottleneck == "unknown"
        assert "rocprofv3" in r.warnings[0]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _roofline() -> RooflineResult:
    return RooflineResult(
        compute_sol_pct=10.0,
        memory_sol_pct=80.0,
        efficiency_pct=80.0,
        at_roofline=False,
        headroom_pct=20.0,
        bottleneck="memory",
        uses_tensor_cores=False,
    )


class TestBottleneckPrompt:
    def test_nvidia_default_unchanged(self):
        prompt = build_bottleneck_prompt(
            "code", {}, _roofline(), {"name": "NVIDIA H100", "sm_count": 132}
        )
        assert "Analyze the NCU metrics" in prompt
        assert "## NCU Metrics" in prompt
        assert "- Tensor Cores: No" in prompt
        assert "- SM Count: 132\n\n## Roofline Analysis" in prompt

    def test_rocm_prompt(self):
        specs = {
            "name": "AMD Instinct MI300X",
            "vendor": "AMD",
            "sm_count": 304,
            "wavefront_size": 64,
        }
        prompt = build_bottleneck_prompt(
            "code",
            {"memory_sol_pct": 80.0, "hbm_bandwidth_gbps": 4000.0},
            _roofline(),
            specs,
            profiler_name="rocprofv3",
            metric_sections=ROCPROF_METRIC_SECTIONS,
            matrix_unit_name="Matrix Cores (MFMA/WMMA)",
            hardware_notes=amd_hardware_notes(specs),
        )
        assert "## rocprofv3 Metrics" in prompt
        assert "- CU Count: 304" in prompt
        assert "- Wavefront Size: 64 threads" in prompt
        assert "num_warps * 64" in prompt
        assert "- Matrix Cores (MFMA/WMMA): No" in prompt
        assert "HBM Bandwidth: 4e+03 GB/s" in prompt


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

_ASM = """\
\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
\t.text
add_kernel:
\t.file\t1 "{path}" "k.py"
\t.loc\t1 3 0 prologue_end ; k.py:3:0
\tglobal_load_dword v1, v0, s[0:1] ; comment {path}
\t{op} v2, v1, v1
\ts_endpgm
\t.section\t.debug_info,"",@progbits
\t.byte\t{dbg} ; DW_TAG
\t.section\t.AMDGPU.csdata,"",@progbits
"""


class TestAmdgcnFingerprint:
    def test_debug_info_ignored(self):
        a = _ASM.format(path="/tmp/a", op="v_add_f32", dbg=1)
        b = _ASM.format(path="/work/b", op="v_add_f32", dbg=7)
        assert normalize_amdgcn(a) == normalize_amdgcn(b)
        assert ".debug" not in normalize_amdgcn(a)

    def test_instruction_change_detected(self):
        a = _ASM.format(path="/tmp/a", op="v_add_f32", dbg=1)
        b = _ASM.format(path="/tmp/a", op="v_mul_f32", dbg=1)
        assert normalize_amdgcn(a) != normalize_amdgcn(b)

    def test_hash_from_amd_cache(self, tmp_path):
        d1, d2 = tmp_path / "c1" / "HASH1", tmp_path / "c2" / "HASH2"
        d1.mkdir(parents=True)
        d2.mkdir(parents=True)
        (d1 / "add_kernel.amdgcn").write_text(
            _ASM.format(path="/x", op="v_add_f32", dbg=1)
        )
        (d2 / "add_kernel.amdgcn").write_text(
            _ASM.format(path="/y", op="v_add_f32", dbg=2)
        )
        h1 = ptx_hash_from_cache(tmp_path / "c1")
        assert h1 is not None
        assert h1 == ptx_hash_from_cache(tmp_path / "c2")


# ---------------------------------------------------------------------------
# Registry / runner / GPU detection
# ---------------------------------------------------------------------------


class TestRocmRegistry:
    def test_all_components_registered(self):
        for component in registry.list_components():
            assert registry.has(component, "rocm"), component

    def test_worker_runner_uses_hip_pinning(self):
        runner = registry.create(
            "worker_runner",
            "rocm",
            log_dir=Path("."),
            logger=None,
            benchmark_lock=None,
            profiling_semaphore=None,
            openai_model="m",
            high_reasoning_effort=False,
            bottleneck_override=None,
            worker_kwargs={},
        )
        assert isinstance(runner, RocmWorkerRunner)
        assert runner._get_worker_target() is _rocm_worker_process

    def test_detect_gpus_prefers_hip_env(self, monkeypatch):
        from triton_kernel_agent.opt_manager import _detect_gpus

        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert _detect_gpus() == [4, 5]

    def test_detect_gpus_from_kfd(self, fake_kfd, monkeypatch):
        from triton_kernel_agent import opt_manager

        monkeypatch.setenv("PATH", "")  # no nvidia-smi
        assert opt_manager._detect_gpus() == [0, 1]
