"""
Bash Gym System Info Service - GPU, VRAM, and RAM detection

Provides hardware detection for:
- GPU information (NVIDIA, AMD, Intel)
- VRAM usage and availability
- System RAM
- CUDA availability
- Python environment
"""

import subprocess
import platform
import psutil
import time
import logging
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    """GPU information."""
    vendor: str           # 'NVIDIA', 'AMD', 'Intel', 'Apple'
    model: str            # 'RTX 4090', 'RX 7900 XT', etc.
    vram: float           # VRAM in GB (0 if unknown)
    vram_used: Optional[float] = None  # Used VRAM in GB
    driver: Optional[str] = None
    temperature: Optional[float] = None  # GPU temperature in Celsius
    utilization: Optional[float] = None  # GPU utilization percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor,
            "model": self.model,
            "vram": self.vram,
            "vram_used": self.vram_used,
            "driver": self.driver,
            "temperature": self.temperature,
            "utilization": self.utilization,
        }


@dataclass
class SystemInfo:
    """Full system information."""
    gpus: List[GpuInfo] = field(default_factory=list)
    total_ram: float = 0.0        # RAM in GB
    available_ram: float = 0.0    # Available RAM in GB
    platform_name: str = ""       # 'win32', 'darwin', 'linux'
    arch: str = ""                # 'x64', 'arm64'
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    python_available: bool = True
    python_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpus": [g.to_dict() for g in self.gpus],
            "total_ram": self.total_ram,
            "available_ram": self.available_ram,
            "platform": self.platform_name,
            "arch": self.arch,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "python_available": self.python_available,
            "python_version": self.python_version,
        }


class SystemInfoService:
    """Service for detecting system hardware information."""

    def __init__(self):
        self._cached_info: Optional[SystemInfo] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 30.0  # 30 seconds cache

    def get_system_info(self, force_refresh: bool = False) -> SystemInfo:
        """Get full system information with caching."""
        now = time.time()

        if not force_refresh and self._cached_info and (now - self._cache_timestamp) < self._cache_ttl:
            logger.debug("Returning cached system info")
            return self._cached_info

        logger.info("Detecting system hardware...")
        try:
            gpus = self.get_gpus()
        except Exception as e:
            logger.error(f"GPU detection failed with exception: {e}")
            gpus = [GpuInfo(vendor="Unknown", model="Detection failed", vram=0.0)]

        try:
            memory = self._get_memory_info()
        except Exception as e:
            logger.error(f"Memory detection failed: {e}")
            memory = {"total": 0.0, "available": 0.0}

        try:
            cuda_info = self._get_cuda_info()
        except Exception as e:
            logger.error(f"CUDA detection failed: {e}")
            cuda_info = {"available": False}

        try:
            python_info = self._get_python_info()
        except Exception as e:
            logger.error(f"Python detection failed: {e}")
            python_info = {"available": True, "version": "unknown"}

        self._cached_info = SystemInfo(
            gpus=gpus,
            total_ram=memory["total"],
            available_ram=memory["available"],
            platform_name=platform.system().lower(),
            arch=platform.machine(),
            cuda_available=cuda_info["available"],
            cuda_version=cuda_info.get("version"),
            python_available=python_info["available"],
            python_version=python_info.get("version"),
        )
        self._cache_timestamp = now

        logger.info(f"System info detected: {len(gpus)} GPU(s), {memory['total']:.1f}GB RAM, CUDA: {cuda_info['available']}")
        return self._cached_info

    def get_gpus(self) -> List[GpuInfo]:
        """Get GPU information."""
        logger.debug("Starting GPU detection...")
        gpus: List[GpuInfo] = []

        # Try NVIDIA first (most common for ML)
        nvidia_gpus = self._get_nvidia_gpus()
        if nvidia_gpus:
            gpus.extend(nvidia_gpus)
            logger.info(f"Found {len(nvidia_gpus)} NVIDIA GPU(s)")

        # If no NVIDIA, try other methods
        if not gpus:
            logger.debug("No NVIDIA GPUs found, trying other detection methods...")

            # Try Windows WMI
            if platform.system() == "Windows":
                logger.debug("Trying Windows WMI detection...")
                wmi_gpus = self._get_windows_gpus()
                if wmi_gpus:
                    gpus.extend(wmi_gpus)
                    logger.info(f"Found {len(wmi_gpus)} GPU(s) via Windows WMI")

            # Try AMD ROCm
            amd_gpus = self._get_amd_gpus()
            if amd_gpus:
                gpus.extend(amd_gpus)
                logger.info(f"Found {len(amd_gpus)} AMD GPU(s)")

        # Fallback
        if not gpus:
            logger.warning("No GPUs detected by any method")
            gpus.append(GpuInfo(
                vendor="Unknown",
                model="No GPU detected",
                vram=0.0,
            ))

        logger.debug(f"GPU detection complete. Found {len(gpus)} GPU(s)")
        return gpus

    def _get_nvidia_gpus(self) -> List[GpuInfo]:
        """Get NVIDIA GPU info via nvidia-smi or pynvml."""
        gpus: List[GpuInfo] = []

        # Try pynvml first (more reliable library-based approach)
        gpus = self._get_nvidia_via_pynvml()
        if gpus:
            logger.debug(f"Got {len(gpus)} NVIDIA GPU(s) via pynvml")
            return gpus

        # Fall back to nvidia-smi CLI
        logger.debug("Trying nvidia-smi for NVIDIA GPU detection...")
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        name, total_mem, used_mem, util, temp, driver = parts[:6]
                        try:
                            vram_gb = float(total_mem) / 1024  # MiB to GiB
                            vram_used_gb = float(used_mem) / 1024
                            utilization = float(util) if util != "[N/A]" else None
                            temperature = float(temp) if temp != "[N/A]" else None
                        except ValueError as e:
                            logger.debug(f"Could not parse nvidia-smi values: {e}")
                            vram_gb = 0.0
                            vram_used_gb = None
                            utilization = None
                            temperature = None

                        gpus.append(GpuInfo(
                            vendor="NVIDIA",
                            model=name,
                            vram=round(vram_gb, 1),
                            vram_used=round(vram_used_gb, 1) if vram_used_gb else None,
                            driver=driver if driver != "[N/A]" else None,
                            temperature=temperature,
                            utilization=utilization,
                        ))
                        logger.debug(f"Found NVIDIA GPU via nvidia-smi: {name} ({vram_gb:.1f}GB)")
            else:
                logger.debug(f"nvidia-smi returned no data. Return code: {result.returncode}")
                if result.stderr:
                    logger.debug(f"stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out after 10 seconds")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found in PATH")
        except Exception as e:
            logger.warning(f"nvidia-smi detection failed: {e}")

        return gpus

    def _get_nvidia_via_pynvml(self) -> List[GpuInfo]:
        """Get NVIDIA GPU info via pynvml library (more reliable)."""
        gpus: List[GpuInfo] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_gb = mem_info.total / (1024 ** 3)  # bytes to GB
                vram_used_gb = mem_info.used / (1024 ** 3)

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except pynvml.NVMLError:
                    utilization = None

                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    temperature = None

                try:
                    driver = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver, bytes):
                        driver = driver.decode('utf-8')
                except pynvml.NVMLError:
                    driver = None

                gpus.append(GpuInfo(
                    vendor="NVIDIA",
                    model=name,
                    vram=round(vram_gb, 1),
                    vram_used=round(vram_used_gb, 1),
                    driver=driver,
                    temperature=temperature,
                    utilization=utilization,
                ))
                logger.debug(f"Found NVIDIA GPU via pynvml: {name} ({vram_gb:.1f}GB)")

            pynvml.nvmlShutdown()
        except ImportError:
            logger.debug("pynvml not installed, falling back to nvidia-smi")
        except Exception as e:
            logger.debug(f"pynvml detection failed: {e}")

        return gpus

    def _get_amd_gpus(self) -> List[GpuInfo]:
        """Get AMD GPU info via rocm-smi."""
        gpus: List[GpuInfo] = []
        logger.debug("Trying AMD ROCm detection...")
        try:
            # Get GPU name
            name_result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Get VRAM info
            mem_result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if name_result.returncode == 0 and "GPU" in name_result.stdout:
                # Parse GPU names
                import re
                gpu_names = re.findall(r"GPU\[\d+\]\s+:\s+Card series:\s+(.+)", name_result.stdout)
                if not gpu_names:
                    gpu_names = ["AMD GPU (ROCm)"]

                # Try to parse VRAM from meminfo output
                vram_gb = 0.0
                if mem_result.returncode == 0:
                    # Look for "VRAM Total Memory" line
                    total_match = re.search(r"VRAM Total Memory.*?(\d+)\s*(?:MB|MiB)", mem_result.stdout, re.IGNORECASE)
                    if total_match:
                        vram_gb = float(total_match.group(1)) / 1024
                    else:
                        # Try GB format
                        total_match = re.search(r"VRAM Total Memory.*?(\d+(?:\.\d+)?)\s*(?:GB|GiB)", mem_result.stdout, re.IGNORECASE)
                        if total_match:
                            vram_gb = float(total_match.group(1))

                for name in gpu_names:
                    gpus.append(GpuInfo(
                        vendor="AMD",
                        model=name.strip(),
                        vram=round(vram_gb, 1),
                    ))
                    logger.debug(f"Found AMD GPU via ROCm: {name.strip()} ({vram_gb:.1f}GB)")
        except subprocess.TimeoutExpired:
            logger.debug("rocm-smi timed out")
        except FileNotFoundError:
            logger.debug("rocm-smi not found in PATH")
        except Exception as e:
            logger.debug(f"AMD ROCm detection failed: {e}")

        return gpus

    def _get_windows_gpus(self) -> List[GpuInfo]:
        """Get GPU info on Windows via WMI."""
        gpus: List[GpuInfo] = []
        try:
            logger.debug("Querying Windows WMI for GPU info...")
            # Use PowerShell to query WMI - increased timeout for slower systems
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",  # Skip profile loading for faster startup
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json"
                ],
                capture_output=True,
                text=True,
                timeout=15,  # Increased from 5s - PowerShell startup can be slow
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WMI JSON output: {e}")
                    logger.debug(f"Raw output was: {result.stdout[:500]}")
                    return gpus

                if not isinstance(data, list):
                    data = [data]

                for controller in data:
                    name = controller.get("Name", "Unknown GPU")
                    # Skip virtual/basic adapters but log it
                    if any(skip in name.lower() for skip in ["microsoft", "basic", "virtual"]):
                        logger.debug(f"Skipping virtual adapter: {name}")
                        continue

                    adapter_ram = controller.get("AdapterRAM")
                    vram_gb = 0.0
                    if adapter_ram:
                        try:
                            vram_gb = round(int(adapter_ram) / (1024 ** 3), 1)
                        except (ValueError, TypeError):
                            logger.debug(f"Could not parse AdapterRAM: {adapter_ram}")

                    vendor = self._detect_vendor(name)

                    gpus.append(GpuInfo(
                        vendor=vendor,
                        model=name,
                        vram=vram_gb,
                        driver=controller.get("DriverVersion"),
                    ))
                    logger.debug(f"Found GPU via WMI: {vendor} {name} ({vram_gb}GB)")
            else:
                logger.debug(f"WMI query returned no data or failed. Return code: {result.returncode}")
                if result.stderr:
                    logger.debug(f"stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("PowerShell WMI query timed out after 15 seconds")
        except FileNotFoundError:
            logger.debug("PowerShell not found")
        except Exception as e:
            logger.warning(f"Windows GPU detection failed: {e}")

        return gpus

    def _detect_vendor(self, name: str) -> str:
        """Detect GPU vendor from model name."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"]):
            return "NVIDIA"
        elif any(kw in name_lower for kw in ["amd", "radeon", "rx ", "vega"]):
            return "AMD"
        elif any(kw in name_lower for kw in ["intel", "arc ", "iris", "uhd"]):
            return "Intel"
        elif "apple" in name_lower:
            return "Apple"
        return "Unknown"

    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory info."""
        try:
            mem = psutil.virtual_memory()
            return {
                "total": round(mem.total / (1024 ** 3), 1),  # Convert to GB
                "available": round(mem.available / (1024 ** 3), 1),
            }
        except Exception:
            return {"total": 0.0, "available": 0.0}

    def _get_cuda_info(self) -> Dict[str, Any]:
        """Detect CUDA availability and version."""
        # Try nvidia-smi first (indicates CUDA driver is installed)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # nvidia-smi works, try to get CUDA version
                try:
                    nvcc_result = subprocess.run(
                        ["nvcc", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if nvcc_result.returncode == 0:
                        import re
                        match = re.search(r"release (\d+\.\d+)", nvcc_result.stdout)
                        if match:
                            return {"available": True, "version": match.group(1)}
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

                # nvidia-smi works but nvcc not found - CUDA likely available via PyTorch
                return {"available": True}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try PyTorch CUDA detection as fallback
        try:
            import torch
            if torch.cuda.is_available():
                version = torch.version.cuda or "unknown"
                return {"available": True, "version": version}
        except ImportError:
            pass

        # Try ROCm (AMD)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showdriverversion"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return {"available": True, "version": "ROCm"}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return {"available": False}

    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python version info."""
        import sys
        return {
            "available": True,
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }

    def clear_cache(self) -> None:
        """Clear cached system info."""
        self._cached_info = None
        self._cache_timestamp = 0

    def get_model_recommendations(self) -> Dict[str, Any]:
        """Get model recommendations based on available VRAM."""
        info = self.get_system_info()
        max_vram = max((g.vram for g in info.gpus), default=0)

        recommendations = {
            "max_vram_gb": max_vram,
            "cuda_available": info.cuda_available,
            "recommended_models": [],
            "recommended_quantization": "4bit",
            "recommended_batch_size": 1,
            "warning": None,
        }

        if max_vram >= 24:
            recommendations["recommended_models"] = [
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "Qwen/Qwen2.5-Coder-3B-Instruct",
            ]
            recommendations["recommended_batch_size"] = 4
            recommendations["recommended_quantization"] = "4bit"
        elif max_vram >= 12:
            recommendations["recommended_models"] = [
                "Qwen/Qwen2.5-Coder-3B-Instruct",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            ]
            recommendations["recommended_batch_size"] = 2
            recommendations["recommended_quantization"] = "4bit"
        elif max_vram >= 8:
            recommendations["recommended_models"] = [
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
            ]
            recommendations["recommended_batch_size"] = 1
            recommendations["recommended_quantization"] = "4bit"
        elif max_vram >= 4:
            recommendations["recommended_models"] = [
                "meta-llama/Llama-3.2-1B-Instruct",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ]
            recommendations["recommended_batch_size"] = 1
            recommendations["recommended_quantization"] = "4bit"
            recommendations["warning"] = "Limited VRAM - use small models with QLoRA"
        else:
            recommendations["recommended_models"] = []
            recommendations["warning"] = "Insufficient VRAM for local training. Consider cloud training or CPU-only inference."

        return recommendations


# Singleton instance
_system_info_service: Optional[SystemInfoService] = None


def get_system_info_service() -> SystemInfoService:
    """Get the singleton SystemInfoService instance."""
    global _system_info_service
    if _system_info_service is None:
        _system_info_service = SystemInfoService()
    return _system_info_service
