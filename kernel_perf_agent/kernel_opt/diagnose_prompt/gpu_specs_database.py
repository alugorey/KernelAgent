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
GPU Specifications Database - Updated with Specific SKUs

This module contains the GPU hardware specifications database used for
performance analysis and bottleneck identification. Updated to include
specific SKU variants for multi-SKU GPUs like A100 and H100.

Sources:
- NVIDIA official specifications and datasheets
- TechPowerUp GPU Database
- Manufacturer datasheets

Last Updated: January 2026
"""

from types import MappingProxyType

_GPU_SPECS_DATABASE: dict[str, dict[str, object]] = {
    # NVIDIA A100 SKUs - SXM4 Variants
    "NVIDIA A100 SXM4 40GB": {
        "name": "NVIDIA A100 SXM4 40GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 1555,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 40,
        "memory_type": "HBM2e",
        "form_factor": "SXM4",
        "tdp_w": 400,
    },
    "NVIDIA A100 SXM4 80GB": {
        "name": "NVIDIA A100 SXM4 80GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 2039,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "SXM4",
        "tdp_w": 400,
    },
    # NVIDIA A100 SKUs - PCIe Variants
    "NVIDIA A100 PCIe 40GB": {
        "name": "NVIDIA A100 PCIe 40GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 1555,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 40,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 250,
    },
    "NVIDIA A100 PCIe 80GB": {
        "name": "NVIDIA A100 PCIe 80GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 1935,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 300,
    },
    # NVIDIA H100 SKUs - SXM5 Variant
    "NVIDIA H100 SXM5 80GB": {
        "name": "NVIDIA H100 SXM5 80GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 67.0,
        "peak_fp16_tflops": 1979.0,  # Without sparsity
        "peak_bf16_tflops": 1979.0,  # Without sparsity
        "peak_memory_bw_gbps": 3350,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM3",
        "form_factor": "SXM5",
        "tdp_w": 700,
    },
    # NVIDIA H100 SKUs - PCIe Variant
    "NVIDIA H100 PCIe 80GB": {
        "name": "NVIDIA H100 PCIe 80GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 51.0,
        "peak_fp16_tflops": 1513.0,  # Without sparsity
        "peak_bf16_tflops": 1513.0,  # Without sparsity
        "peak_memory_bw_gbps": 2000,
        "sm_count": 114,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 350,
    },
    # NVIDIA H100 SKUs - NVL Variant (for LLM inference)
    "NVIDIA H100 NVL 94GB": {
        "name": "NVIDIA H100 NVL 94GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 60.0,
        "peak_fp16_tflops": 1671.0,  # Without sparsity
        "peak_bf16_tflops": 1671.0,  # Without sparsity
        "peak_memory_bw_gbps": 3900,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 94,
        "memory_type": "HBM3",
        "form_factor": "PCIe",
        "tdp_w": 400,
    },
    # NVIDIA RTX 4090
    "NVIDIA RTX 4090": {
        "name": "NVIDIA RTX 4090",
        "architecture": "Ada Lovelace",
        "peak_fp32_tflops": 82.58,
        "peak_fp16_tflops": 82.58,
        "peak_bf16_tflops": 82.58,
        "peak_memory_bw_gbps": 1008,
        "sm_count": 128,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 72,
        "memory_gb": 24,
        "memory_type": "GDDR6X",
        "form_factor": "PCIe",
        "tdp_w": 450,
    },
    # NVIDIA RTX 5080
    "NVIDIA RTX 5080": {
        "name": "NVIDIA RTX 5080",
        "architecture": "Blackwell",
        "peak_fp32_tflops": 56.28,
        "peak_fp16_tflops": 56.28,
        "peak_bf16_tflops": 56.28,
        "peak_memory_bw_gbps": 960,
        "sm_count": 84,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 64,
        "memory_gb": 16,
        "memory_type": "GDDR7",
        "form_factor": "PCIe",
        "tdp_w": 360,
    },
    # ------------------------------------------------------------------
    # AMD GPUs.  Keys match torch.cuda.get_device_name() on ROCm.
    # "sm_count" holds the Compute Unit (CU) count so code that reads
    # sm_count keeps working; prompts relabel it as CU for vendor == AMD.
    # Peak TFLOPS are dense (no sparsity); FP32 is the vector-ALU rate.
    # Sources: AMD Instinct / Radeon product datasheets.
    # ------------------------------------------------------------------
    # AMD Instinct MI300X (CDNA3, wave64)
    "AMD Instinct MI300X": {
        "name": "AMD Instinct MI300X",
        "vendor": "AMD",
        "architecture": "CDNA3",
        "gfx_arch": "gfx942",
        "wavefront_size": 64,
        "peak_fp32_tflops": 81.7,
        "peak_fp16_tflops": 1307.4,
        "peak_bf16_tflops": 1307.4,
        "peak_memory_bw_gbps": 5300,
        "sm_count": 304,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 32,
        "lds_kb": 64,
        "l2_cache_mb": 32,  # 4 MB per XCD x 8 XCDs (plus 256 MB Infinity Cache)
        "memory_gb": 192,
        "memory_type": "HBM3",
        "form_factor": "OAM",
        "tdp_w": 750,
    },
    # AMD Instinct MI325X (CDNA3, wave64)
    "AMD Instinct MI325X": {
        "name": "AMD Instinct MI325X",
        "vendor": "AMD",
        "architecture": "CDNA3",
        "gfx_arch": "gfx942",
        "wavefront_size": 64,
        "peak_fp32_tflops": 81.7,
        "peak_fp16_tflops": 1307.4,
        "peak_bf16_tflops": 1307.4,
        "peak_memory_bw_gbps": 6000,
        "sm_count": 304,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 32,
        "lds_kb": 64,
        "l2_cache_mb": 32,
        "memory_gb": 256,
        "memory_type": "HBM3E",
        "form_factor": "OAM",
        "tdp_w": 1000,
    },
    # AMD Instinct MI300A APU (CDNA3, wave64)
    "AMD Instinct MI300A": {
        "name": "AMD Instinct MI300A",
        "vendor": "AMD",
        "architecture": "CDNA3",
        "gfx_arch": "gfx942",
        "wavefront_size": 64,
        "peak_fp32_tflops": 61.3,
        "peak_fp16_tflops": 980.6,
        "peak_bf16_tflops": 980.6,
        "peak_memory_bw_gbps": 5300,
        "sm_count": 228,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 32,
        "lds_kb": 64,
        "l2_cache_mb": 24,
        "memory_gb": 128,
        "memory_type": "HBM3",
        "form_factor": "APU",
        "tdp_w": 550,
    },
    # AMD Instinct MI355X (CDNA4, wave64)
    "AMD Instinct MI355X": {
        "name": "AMD Instinct MI355X",
        "vendor": "AMD",
        "architecture": "CDNA4",
        "gfx_arch": "gfx950",
        "wavefront_size": 64,
        "peak_fp32_tflops": 157.3,
        "peak_fp16_tflops": 2516.6,
        "peak_bf16_tflops": 2516.6,
        "peak_memory_bw_gbps": 8000,
        "sm_count": 256,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 32,
        "lds_kb": 160,
        "l2_cache_mb": 32,
        "memory_gb": 288,
        "memory_type": "HBM3E",
        "form_factor": "OAM",
        "tdp_w": 1400,
    },
    # AMD Instinct MI350X (CDNA4, wave64)
    "AMD Instinct MI350X": {
        "name": "AMD Instinct MI350X",
        "vendor": "AMD",
        "architecture": "CDNA4",
        "gfx_arch": "gfx950",
        "wavefront_size": 64,
        "peak_fp32_tflops": 144.2,
        "peak_fp16_tflops": 2306.9,
        "peak_bf16_tflops": 2306.9,
        "peak_memory_bw_gbps": 8000,
        "sm_count": 256,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 32,
        "lds_kb": 160,
        "l2_cache_mb": 32,
        "memory_gb": 288,
        "memory_type": "HBM3E",
        "form_factor": "OAM",
        "tdp_w": 1000,
    },
    # AMD Instinct MI250X — ROCm exposes each GCD as its own device, so
    # these are per-GCD numbers (half of the full OAM package).
    "AMD Instinct MI250X": {
        "name": "AMD Instinct MI250X",
        "vendor": "AMD",
        "architecture": "CDNA2",
        "gfx_arch": "gfx90a",
        "wavefront_size": 64,
        "peak_fp32_tflops": 23.9,
        "peak_fp16_tflops": 191.5,
        "peak_bf16_tflops": 191.5,
        "peak_memory_bw_gbps": 1638,
        "sm_count": 110,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 16,
        "lds_kb": 64,
        "l2_cache_mb": 8,
        "memory_gb": 64,
        "memory_type": "HBM2e",
        "form_factor": "OAM (per GCD)",
        "tdp_w": 280,
    },
    # AMD Radeon RX 7900 XTX (RDNA3, wave32)
    "AMD Radeon RX 7900 XTX": {
        "name": "AMD Radeon RX 7900 XTX",
        "vendor": "AMD",
        "architecture": "RDNA3",
        "gfx_arch": "gfx1100",
        "wavefront_size": 32,
        "peak_fp32_tflops": 61.4,
        "peak_fp16_tflops": 122.8,
        "peak_bf16_tflops": 122.8,
        "peak_memory_bw_gbps": 960,
        "sm_count": 96,
        "max_threads_per_sm": 1024,
        "l1_cache_kb": 32,
        "lds_kb": 64,
        "l2_cache_mb": 6,
        "memory_gb": 24,
        "memory_type": "GDDR6",
        "form_factor": "PCIe",
        "tdp_w": 355,
    },
    # AMD Radeon PRO W7900 (RDNA3, wave32)
    "AMD Radeon PRO W7900": {
        "name": "AMD Radeon PRO W7900",
        "vendor": "AMD",
        "architecture": "RDNA3",
        "gfx_arch": "gfx1100",
        "wavefront_size": 32,
        "peak_fp32_tflops": 61.3,
        "peak_fp16_tflops": 122.6,
        "peak_bf16_tflops": 122.6,
        "peak_memory_bw_gbps": 864,
        "sm_count": 96,
        "max_threads_per_sm": 1024,
        "l1_cache_kb": 32,
        "lds_kb": 64,
        "l2_cache_mb": 6,
        "memory_gb": 48,
        "memory_type": "GDDR6",
        "form_factor": "PCIe",
        "tdp_w": 295,
    },
}

# Make database read-only to prevent accidental modification
GPU_SPECS_DATABASE = MappingProxyType(_GPU_SPECS_DATABASE)
