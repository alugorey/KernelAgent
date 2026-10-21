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
Platform configuration registry for multi-backend support.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"

# Triton-on-ROCm uses the 'cuda' torch device string (HIP masquerades as CUDA).
ROCM_DEVICE_STRING = "cuda"


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend."""

    name: str
    device_string: str
    guidance_block: str
    kernel_guidance: str
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)


# Platform-specific constants
_XPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR INTEL XPU:**
- Default tensor allocations to device='xpu' (never 'cuda'); CPU is allowed only when necessary.
- Check availability with: hasattr(torch, 'xpu') and torch.xpu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device
- Do NOT set TRITON_BACKENDS environment variable
- Do NOT import or disable XPUDriver
- Use torch.xpu.synchronize() if synchronization is needed
- Intel XPU subgroup size is typically 16 (not 32 like CUDA warps)
- Preferred block sizes: 64, 128, 256, or 512"""

_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

_XPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
    "from triton.backends.intel.driver import XPUDriver",
    "XPUDriver.is_available = classmethod(lambda cls: False)",
)


def _rocm_wavefront_lines(wavefront_size: int | None) -> tuple[str, str]:
    """Return (short requirement line, detailed guidance) for the wavefront size."""
    if wavefront_size == 64:
        return (
            (
                "- Wavefront size is 64 (CDNA/Instinct): Triton num_warps counts 64-wide "
                "wavefronts, so num_warps=4 is 256 threads"
            ),
            """\
   - This GPU runs **wave64**: one Triton "warp" is a 64-lane wavefront
   - Threads per program = num_warps * 64 (num_warps=4 -> 256 threads, 8 -> 512)
   - Prefer num_warps in {1, 2, 4, 8}; 4 is a good default, 8 for large tiles
   - A per-wavefront reduction spans 64 lanes; never hardcode 32""",
        )
    if wavefront_size == 32:
        return (
            (
                "- Wavefront size is 32 (RDNA/gfx12): Triton num_warps counts 32-wide "
                "wavefronts, so num_warps=4 is 128 threads"
            ),
            """\
   - This GPU runs **wave32**: one Triton "warp" is a 32-lane wavefront
   - Threads per program = num_warps * 32 (num_warps=4 -> 128 threads, 8 -> 256)
   - Prefer num_warps in {2, 4, 8, 16}; use roughly 2x the num_warps you would on wave64
   - Matrix ops use WMMA, not MFMA; MFMA-only knobs (matrix_instr_nonkdim, kpack) do not apply""",
        )
    return (
        (
            "- Wavefront size is 32 or 64 depending on the GPU; query it with "
            "torch.cuda.get_device_properties(0).warp_size and never hardcode 32"
        ),
        """\
   - Wavefront size depends on the GPU: 64 on CDNA/Instinct (gfx9xx), 32 on RDNA/gfx12
   - Threads per program = num_warps * wavefront_size
   - Read it at runtime: torch.cuda.get_device_properties(0).warp_size
   - Write reductions/layouts that are correct for both 32 and 64""",
    )


def make_rocm_platform(wavefront_size: int | None) -> PlatformConfig:
    """Build the ROCm PlatformConfig for a given wavefront size (32, 64, or None)."""
    short_line, detail = _rocm_wavefront_lines(wavefront_size)
    guidance_block = f"""\
**CRITICAL PLATFORM REQUIREMENTS FOR AMD ROCm:**
- ROCm PyTorch uses device='cuda' (HIP is exposed through the torch.cuda API); never use 'hip'
- Use torch.cuda.synchronize() / torch.cuda.Event as usual; they map to HIP
- Detect ROCm with torch.version.hip is not None if behaviour must differ from NVIDIA
- Do NOT emit NVIDIA-only features: inline PTX, TMA / tensor descriptors, wgmma, warp_specialize
{short_line}"""

    kernel_guidance = f"""\
## AMD ROCm-Specific Optimizations

You are generating a Triton kernel for an AMD GPU via ROCm. Follow these guidelines:

1. **Device Context**: Use device='cuda' (ROCm PyTorch exposes HIP as 'cuda')
2. **Wavefronts**:
{detail}
3. **Memory Hierarchy**:
   - LDS (shared memory) is 64 KB per CU on most parts (160 KB on gfx950/MI355X)
   - Large num_stages multiplies LDS use; num_stages=2 is the usual GEMM default on AMD
   - Coalesce global loads per wavefront; 128-bit vectorized loads/stores are fastest
4. **Matrix Cores**: tl.dot lowers to MFMA (CDNA) or WMMA (RDNA/gfx12); keep BLOCK_K a multiple of 16 (32 for fp8)
5. **AMD-only kernel kwargs** (pass alongside num_warps/num_stages, ignore on NVIDIA):
   - waves_per_eu: occupancy hint (e.g. 2-4) trading registers for more resident wavefronts
   - matrix_instr_nonkdim: 16 or 32 to pick the MFMA tile shape (CDNA only)
6. **Unsupported on AMD**: TMA / tl.make_tensor_descriptor, warp_specialize, inline PTX (tl.inline_asm_elementwise with PTX), cp.async
7. **FP8**: gfx942 (MI300) uses float8_e4m3fnuz / float8_e5m2fnuz; gfx950 and gfx12 use OCP float8_e4m3fn / float8_e5m2"""

    return PlatformConfig(
        name="rocm",
        device_string=ROCM_DEVICE_STRING,
        guidance_block=guidance_block,
        kernel_guidance=kernel_guidance,
        cuda_hacks_to_strip=(),
    )


def _detect_rocm_wavefront_size() -> int | None:
    """Wavefront size of the local AMD GPU without initializing HIP."""
    try:
        from triton_kernel_agent.platform.amd_topology import detect_wavefront_size

        return detect_wavefront_size()
    except Exception as e:
        # Never break import of this module (used by every platform) over a
        # bad override; fall back to wavefront-agnostic guidance.
        import logging

        logging.getLogger(__name__).warning(f"ROCm wavefront detection failed: {e}")
        return None


# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
    # Wavefront size (32 vs 64) is detected from the local GPU, or forced
    # with KERNELAGENT_WAVEFRONT_SIZE when targeting a different GPU.
    "rocm": make_rocm_platform(_detect_rocm_wavefront_size()),
}


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
