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
rocprofv3 Profiling Module for Triton Kernels on AMD GPUs

The AMD counterpart of ``ncu_profiler.py``.  It wraps three tasks:
1) Collect hardware counters with rocprofv3, one application run per
   counter pass (``profile_triton_kernel_rocprof``).
2) Load the per-pass CSVs and keep the last dispatch of every kernel
   (``load_rocprof_counters``).
3) Derive NCU-like, LLM-friendly metrics (SOL %, bandwidth, hit rates,
   occupancy, stalls) from the raw counters (``derive_rocprof_metrics``).

Hardware can only collect a limited number of counters per pass (e.g.
FETCH_SIZE and WRITE_SIZE cannot share a pass on gfx942), so counters are
grouped into ``ROCPROF_COUNTER_PASSES``.  Counters the GPU does not expose
(names differ between CDNA and RDNA) are dropped up front, and a pass that
still exceeds the hardware limits is split in half and retried.

Kernel durations are measured under counter collection (kernels are
serialized and instrumented), so they run slightly slower than in the
benchmark; use them for bandwidth ratios, not for timing.
"""

from __future__ import annotations

import functools
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# Counter groups, one rocprofv3 application run each.  Validated to fit in a
# single pass on gfx942 (MI300X).
ROCPROF_COUNTER_PASSES: list[list[str]] = [
    ["FETCH_SIZE", "GRBM_GUI_ACTIVE", "SQ_WAVES", "MAX_WAVE_SIZE"],
    ["WRITE_SIZE", "TCC_HIT_sum", "TCC_MISS_sum"],
    [
        "VALUBusy",
        "MfmaUtil",
        "SALUBusy",
        "MeanOccupancyPerActiveCU",
        "OccupancyPercent",
    ],
    [
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
        "SQ_INSTS_VMEM_RD",
        "SQ_INSTS_VMEM_WR",
        "SQ_INSTS_LDS",
        "SQ_INSTS_MFMA",
        "SQ_INSTS_BRANCH",
    ],
    [
        "SQ_WAVE_CYCLES",
        "SQ_WAIT_INST_ANY",
        "SQ_WAIT_INST_LDS",
        "LDSBankConflict",
        "MemUnitStalled",
    ],
    [
        "TCP_TOTAL_CACHE_ACCESSES_sum",
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
        "TCC_BUSY_avr",
        "GRBM_GUI_ACTIVE",
    ],
]

# Per-dispatch columns rocprofv3 writes next to every counter value.
_DISPATCH_COLUMNS = (
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
)

# Prefixes of PyTorch-internal kernels (same convention as the NCU path).
_PYTORCH_KERNEL_PREFIXES = ("at::", "void at::")

_MAX_KERNEL_NAME_LEN = 200

_HW_LIMIT_MARKERS = (
    "exceeds the capabilities of the hardware",
    "could not construct profile cfg",
)


class RocprofError(RuntimeError):
    """rocprofv3 failed to collect counters."""


def find_rocprofv3(rocprof_bin: str | None = None) -> str:
    """Resolve the rocprofv3 executable or raise ``FileNotFoundError``."""
    candidates = [
        rocprof_bin,
        os.environ.get("KERNELAGENT_ROCPROFV3"),
        shutil.which("rocprofv3"),
        "/opt/rocm/bin/rocprofv3",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        "rocprofv3 not found; install ROCm profiler SDK or set KERNELAGENT_ROCPROFV3"
    )


def rocprof_version(rocprof_bin: str) -> str | None:
    """Return the rocprofv3 version string (e.g. ``"1.4.2"``), if available."""
    try:
        result = subprocess.run(
            [rocprof_bin, "--version"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    m = re.search(r"version:\s*(\S+)", result.stdout + result.stderr)
    return m.group(1) if m else None


@functools.lru_cache(maxsize=4)
def list_available_counters(rocprof_bin: str) -> frozenset[str]:
    """Counters exposed by the first GPU (``rocprofv3 -L``), cached per process.

    Returns an empty set if listing fails, which callers treat as "unknown"
    (no filtering).
    """
    try:
        result = subprocess.run(
            [rocprof_bin, "-L"], capture_output=True, text=True, timeout=60
        )
    except Exception:
        return frozenset()
    return parse_counter_listing(result.stdout)


def parse_counter_listing(text: str) -> frozenset[str]:
    """Parse ``rocprofv3 -L`` output, keeping the first ``GPU:<n>`` section."""
    counters: set[str] = set()
    in_first_gpu = False
    for line in text.splitlines():
        if re.match(r"^GPU:\d+\s*$", line.strip()):
            if in_first_gpu:
                break
            in_first_gpu = True
            continue
        if in_first_gpu:
            m = re.match(r"^\s*Counter_Name\s*:\s*(\S+)", line)
            if m:
                counters.add(m.group(1))
    return frozenset(counters)


def filter_passes(
    passes: Sequence[Sequence[str]], available: frozenset[str]
) -> list[list[str]]:
    """Drop counters the GPU does not expose; drop passes left empty."""
    if not available:
        return [list(p) for p in passes]
    filtered = [[c for c in p if c in available] for p in passes]
    return [p for p in filtered if p]


def _run_pass(
    rocprof_bin: str,
    counters: list[str],
    cmd_tail: list[str],
    out_dir: Path,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> Path:
    """Run one rocprofv3 counter pass; return its counter_collection CSV."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        rocprof_bin,
        "--pmc",
        *counters,
        "--output-format",
        "csv",
        "-d",
        str(out_dir),
        "-o",
        "pass",
        "--",
        *cmd_tail,
    ]
    result = subprocess.run(
        cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout
    )
    csvs = sorted(out_dir.rglob("*counter_collection.csv"))
    if result.returncode != 0 or not csvs:
        msg = (result.stderr or "") + (result.stdout or "")
        raise RocprofError(
            f"rocprofv3 pass {counters} failed (rc={result.returncode}): "
            f"{msg.strip()[-500:]}"
        )
    return csvs[0]


def _run_pass_with_split(
    rocprof_bin: str,
    counters: list[str],
    cmd_tail: list[str],
    out_dir: Path,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    log: list[str],
) -> list[Path]:
    """Run a pass; on a hardware-limit failure, split it in half and retry."""
    try:
        return [_run_pass(rocprof_bin, counters, cmd_tail, out_dir, cwd, env, timeout)]
    except RocprofError as e:
        msg = str(e).lower()
        hw_limit = any(m in msg for m in _HW_LIMIT_MARKERS)
        if len(counters) == 1 or not hw_limit:
            log.append(f"dropped {counters}: {str(e)[:200]}")
            return []
        mid = len(counters) // 2
        log.append(f"split {counters} (hardware counter limit)")
        return _run_pass_with_split(
            rocprof_bin,
            counters[:mid],
            cmd_tail,
            out_dir.with_name(out_dir.name + "a"),
            cwd,
            env,
            timeout,
            log,
        ) + _run_pass_with_split(
            rocprof_bin,
            counters[mid:],
            cmd_tail,
            out_dir.with_name(out_dir.name + "b"),
            cwd,
            env,
            timeout,
            log,
        )


def profile_triton_kernel_rocprof(
    benchmark_script: Path,
    workdir: Path,
    out_subdir: str = "rocprof_output",
    python_executable: str | None = None,
    rocprof_bin: str | None = None,
    passes: Sequence[Sequence[str]] = ROCPROF_COUNTER_PASSES,
    timeout: int = 360,
) -> list[Path]:
    """
    Profile a Triton kernel with rocprofv3, one application run per pass.

    Args:
        benchmark_script: Script that launches the kernel (warmup + 1 run).
        workdir: Working directory; outputs go to ``workdir/out_subdir``.
        out_subdir: Output directory name under *workdir*.
        python_executable: Python to run the script (default: sys.executable,
            overridable via KERNEL_PROFILER_PYTHON like the NCU path).
        rocprof_bin: Path to rocprofv3 (default: auto-detect).
        passes: Counter groups; each group is one rocprofv3 run.
        timeout: Timeout in seconds for *each* pass.

    Returns:
        List of counter_collection CSV paths, one per successful pass.

    Raises:
        FileNotFoundError: rocprofv3 or the benchmark script is missing.
        RocprofError: No pass produced any data.
        subprocess.TimeoutExpired: A pass timed out.
    """
    rocprof_bin = find_rocprofv3(rocprof_bin)
    if python_executable is None:
        python_executable = os.environ.get("KERNEL_PROFILER_PYTHON") or sys.executable

    benchmark_script = benchmark_script.resolve()
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")

    out_root = (workdir / out_subdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    # Shared cache: only the first pass pays for Triton compilation.
    env["TRITON_CACHE_DIR"] = str(workdir / ".triton_cache")

    active_passes = filter_passes(passes, list_available_counters(rocprof_bin))
    cmd_tail = [python_executable, str(benchmark_script)]

    print(f"[rocprofv3] Running {len(active_passes)} counter passes...")
    print(f"[rocprofv3] Benchmark: {benchmark_script.name}")
    print(f"[rocprofv3] Output: {out_root}")

    csv_paths: list[Path] = []
    log: list[str] = []
    for i, counters in enumerate(active_passes):
        csv_paths.extend(
            _run_pass_with_split(
                rocprof_bin,
                list(counters),
                cmd_tail,
                out_root / f"pass{i}",
                workdir,
                env,
                timeout,
                log,
            )
        )
    for line in log:
        print(f"[rocprofv3] {line}")

    if not csv_paths:
        raise RocprofError("rocprofv3 produced no counter data: " + "; ".join(log))

    print(f"[rocprofv3] ✓ Collected {len(csv_paths)} pass(es)")
    return csv_paths


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_rocprof_counters(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """Load per-pass counter CSVs into one row per (kernel, pass).

    For every kernel name only the **last** dispatch is kept (the wrapper
    runs warmup iterations first, mirroring NCU's ``select="last"``).

    Columns: ``Kernel Name``, ``pass``, ``dispatch_id``, ``duration_ns``,
    the dispatch resource columns, and one column per counter collected in
    that pass (NaN for counters from other passes).
    """
    rows: list[dict[str, Any]] = []
    for pass_idx, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        last_dispatch = df.groupby("Kernel_Name")["Dispatch_Id"].transform("max")
        df = df[df["Dispatch_Id"] == last_dispatch]
        for (name, dispatch_id), grp in df.groupby(["Kernel_Name", "Dispatch_Id"]):
            first = grp.iloc[0]
            row: dict[str, Any] = {
                "Kernel Name": str(name),
                "pass": pass_idx,
                "dispatch_id": int(dispatch_id),
                "duration_ns": float(first["End_Timestamp"] - first["Start_Timestamp"]),
            }
            for col in _DISPATCH_COLUMNS:
                if col in grp.columns:
                    row[col] = first[col]
            # Some derived counters are reported once per dimension; sum them.
            for counter, value in (
                grp.groupby("Counter_Name")["Counter_Value"].sum().items()
            ):
                row[str(counter)] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def _is_pytorch_kernel(name: str) -> bool:
    return name.startswith(_PYTORCH_KERNEL_PREFIXES)


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pct(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return max(0.0, min(100.0, 100.0 * num / den))


def derive_rocprof_metrics(
    pass_rows: Sequence[dict[str, Any]],
    wavefront_size: int | None = None,
    peak_memory_bw_gbps: float | None = None,
    num_xcc: int = 1,
) -> dict[str, Any]:
    """Turn one kernel's per-pass raw counters into derived metrics.

    Args:
        pass_rows: Rows from :func:`load_rocprof_counters` for one kernel.
        wavefront_size: 32 or 64.  ``MAX_WAVE_SIZE`` from the profile wins
            when present; this is the fallback.
        peak_memory_bw_gbps: Peak HBM bandwidth for the %-of-peak metric.
        num_xcc: Accelerator complex dies (8 on MI300X).  GRBM_GUI_ACTIVE
            is summed across XCDs, so it is divided by this for cycles.

    Returns:
        Dict keyed by the names in ``ROCPROF_METRIC_SECTIONS``.  Metrics whose
        counters were not collected are omitted.
    """

    def get(counter: str) -> float | None:
        for r in pass_rows:
            v = _finite(r.get(counter))
            if v is not None:
                return v
        return None

    def same_pass(*counters: str) -> tuple[dict[str, Any] | None, list[float | None]]:
        """Values of *counters* from a single pass (ratios across passes are noisy)."""
        for r in pass_rows:
            vals = [_finite(r.get(c)) for c in counters]
            if all(v is not None for v in vals):
                return r, vals
        return None, [None] * len(counters)

    m: dict[str, Any] = {}
    first = pass_rows[0] if pass_rows else {}

    profiled_wave = get("MAX_WAVE_SIZE")
    wave = int(profiled_wave) if profiled_wave else wavefront_size
    if wave:
        m["wavefront_size"] = wave

    durations = [r["duration_ns"] for r in pass_rows if r.get("duration_ns")]
    if durations:
        m["duration_us"] = sorted(durations)[len(durations) // 2] / 1000.0

    # --- Memory bandwidth (FETCH_SIZE / WRITE_SIZE are in KB) ------------
    bw_gbps = 0.0
    have_bw = False
    for counter, key in (
        ("FETCH_SIZE", "hbm_read_bytes"),
        ("WRITE_SIZE", "hbm_write_bytes"),
    ):
        row, (kb,) = same_pass(counter)
        if row is None:
            continue
        nbytes = kb * 1024.0
        m[key] = nbytes
        if row.get("duration_ns"):
            bw_gbps += nbytes / row["duration_ns"]  # bytes/ns == GB/s
            have_bw = True
    if have_bw:
        m["hbm_bandwidth_gbps"] = bw_gbps
        if peak_memory_bw_gbps:
            m["hbm_bw_pct_of_peak"] = _pct(bw_gbps, peak_memory_bw_gbps)

    _, (hit, miss) = same_pass("TCC_HIT_sum", "TCC_MISS_sum")
    if hit is not None and (hit + miss) > 0:
        m["l2_hit_rate_pct"] = _pct(hit, hit + miss)

    _, (tcc_busy, gui_active) = same_pass("TCC_BUSY_avr", "GRBM_GUI_ACTIVE")
    if tcc_busy is not None and gui_active:
        m["l2_busy_pct"] = _pct(tcc_busy, gui_active / max(1, num_xcc))

    _, (tcp_total, tcp_rd, tcp_wr) = same_pass(
        "TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum"
    )
    if tcp_total:
        m["l1_hit_rate_pct"] = _pct(tcp_total - tcp_rd - tcp_wr, tcp_total)

    # --- Compute utilization ---------------------------------------------
    for counter, key in (
        ("VALUBusy", "valu_busy_pct"),
        ("MfmaUtil", "mfma_util_pct"),
        ("SALUBusy", "salu_busy_pct"),
        ("OccupancyPercent", "occupancy_pct"),
    ):
        v = get(counter)
        if v is not None:
            m[key] = max(0.0, min(100.0, v))
    v = get("MeanOccupancyPerActiveCU")
    if v is not None:
        m["waves_per_cu"] = v

    for counter, key in (
        ("SQ_WAVES", "total_waves"),
        ("SQ_INSTS_VALU", "valu_insts"),
        ("SQ_INSTS_SALU", "salu_insts"),
        ("SQ_INSTS_VMEM_RD", "vmem_read_insts"),
        ("SQ_INSTS_VMEM_WR", "vmem_write_insts"),
        ("SQ_INSTS_LDS", "lds_insts"),
        ("SQ_INSTS_MFMA", "mfma_insts"),
        ("SQ_INSTS_BRANCH", "branch_insts"),
    ):
        v = get(counter)
        if v is not None:
            m[key] = int(v)

    # --- Stalls ----------------------------------------------------------
    _, (wait_any, wave_cycles) = same_pass("SQ_WAIT_INST_ANY", "SQ_WAVE_CYCLES")
    if wait_any is not None:
        m["dependency_wait_pct"] = _pct(wait_any, wave_cycles)
    _, (wait_lds, wave_cycles) = same_pass("SQ_WAIT_INST_LDS", "SQ_WAVE_CYCLES")
    if wait_lds is not None:
        m["lds_wait_pct"] = _pct(wait_lds, wave_cycles)
    for counter, key in (
        ("MemUnitStalled", "mem_unit_stalled_pct"),
        ("LDSBankConflict", "lds_bank_conflict_pct"),
    ):
        v = get(counter)
        if v is not None:
            m[key] = max(0.0, min(100.0, v))

    # --- Resources (from the dispatch record) ----------------------------
    for col, key in (
        ("Workgroup_Size", "workgroup_size"),
        ("Grid_Size", "grid_size"),
        ("VGPR_Count", "vgpr_count"),
        ("Accum_VGPR_Count", "agpr_count"),
        ("SGPR_Count", "sgpr_count"),
        ("LDS_Block_Size", "lds_bytes_per_workgroup"),
        ("Scratch_Size", "scratch_bytes_per_lane"),
    ):
        v = _finite(first.get(col))
        if v is not None:
            m[key] = int(v)
    if wave and m.get("workgroup_size"):
        m["waves_per_workgroup"] = math.ceil(m["workgroup_size"] / wave)

    # --- Speed of light --------------------------------------------------
    compute = [m[k] for k in ("valu_busy_pct", "mfma_util_pct") if m.get(k) is not None]
    if compute:
        m["compute_sol_pct"] = max(compute)
    # Memory SOL is HBM bandwidth only.  TCC_BUSY counts cycles with *any*
    # L2 request in flight, so a latency-bound kernel with a few streaming
    # wavefronts reads ~50% "busy" at <10% of peak bandwidth; it is reported
    # as l2_busy_pct but is not a throughput SOL.
    if m.get("hbm_bw_pct_of_peak") is not None:
        m["memory_sol_pct"] = m["hbm_bw_pct_of_peak"]

    return m


def rocprof_metrics_by_kernel(
    df: pd.DataFrame,
    wavefront_size: int | None = None,
    peak_memory_bw_gbps: float | None = None,
    num_xcc: int = 1,
    round_digits: int | None = 3,
) -> dict[str, dict[str, Any]]:
    """Derive metrics per kernel, shaped like the NCU path's metrics dict.

    PyTorch-internal kernels are dropped when any Triton kernel is present.
    Kernels are ordered most-recently-dispatched first, so the first entry
    corresponds to NCU's "last profiled kernel".
    """
    if df is None or df.empty:
        return {}

    names = list(df["Kernel Name"].unique())
    triton_names = [n for n in names if not _is_pytorch_kernel(n)]
    keep = triton_names or names

    last_dispatch = df.groupby("Kernel Name")["dispatch_id"].max()
    keep.sort(key=lambda n: last_dispatch[n], reverse=True)

    out: dict[str, dict[str, Any]] = {}
    for name in keep:
        rows = df[df["Kernel Name"] == name].to_dict(orient="records")
        derived = derive_rocprof_metrics(
            rows,
            wavefront_size=wavefront_size,
            peak_memory_bw_gbps=peak_memory_bw_gbps,
            num_xcc=num_xcc,
        )
        if round_digits is not None:
            derived = {
                k: round(v, round_digits) if isinstance(v, float) else v
                for k, v in derived.items()
            }
        out[name[:_MAX_KERNEL_NAME_LEN]] = derived
    return out
