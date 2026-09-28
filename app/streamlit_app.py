"""
SPARTAN — Satellite Pixel-Augmented Resolution & Terrain Analysis Network
Streamlit front-end (v0.1.0)

Enhances 10 m Sentinel-2 GeoTIFFs to <2.5 m while preserving spectral
fidelity and geospatial CRS metadata.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import subprocess
import sys
from datetime import date, datetime

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Optional heavy / UI dependencies — degrade gracefully when absent
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False
    Resampling = None  # type: ignore[assignment, misc]
    MemoryFile = None  # type: ignore[assignment, misc]
    Affine = None  # type: ignore[assignment, misc]

try:
    import torch
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    peak_signal_noise_ratio = None  # type: ignore[assignment]
    structural_similarity_index_measure = None  # type: ignore[assignment]

from spartan.data.preprocessor import preprocess_array

try:
    from importlib import import_module

    _image_comparison = import_module(
        "streamlit_image_comparison"
    ).image_comparison

    HAS_IMAGE_COMPARISON = True
except ImportError:  # pragma: no cover
    HAS_IMAGE_COMPARISON = False
    _image_comparison = None

logger = logging.getLogger("spartan.app")
logging.basicConfig(level=logging.INFO)

APP_VERSION = "0.1.0"
SIH_CONTEXT = "Smart India Hackathon — Geospatial Super-Resolution"
TARGET_SCALE = 4  # 10 m → 2.5 m
SUPPORTED_EXTENSIONS = (".tif", ".tiff")

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class ModelChoice(str, Enum):
    LITE = "SPARTAN-Lite (Bicubic Baseline)"
    SRRESNET = "SPARTAN-SRResNet"
    SWIN = "SPARTAN-SwinTransformer"


class BandMode(str, Enum):
    RGB = "RGB (B4-B3-B2)"
    FALSE_COLOR = "False Color IR (B8-B4-B3)"
    NIR = "Single-band NIR (B8)"
    RED = "Single-band Red (B4)"
    GREEN = "Single-band Green (B3)"
    BLUE = "Single-band Blue (B2)"


@dataclass
class GeoMetadata:
    crs: str
    bounds: Tuple[float, float, float, float]
    band_count: int
    width: int
    height: int
    resolution: Tuple[float, float]
    dtype: str
    transform: Optional[Any] = None
    nodata: Optional[float] = None
    driver: str = "GTiff"
    tags: Dict[str, str] = field(default_factory=dict)

    def upsampled(self, scale: int) -> "GeoMetadata":
        """Return metadata reflecting a uniform scale-factor upsample."""
        if self.transform is not None and HAS_RASTERIO:
            # Scale the affine transform so pixel size shrinks by 1/scale
            new_transform = self.transform * Affine.scale(1 / scale, 1 / scale)
        else:
            new_transform = None

        res_x, res_y = self.resolution
        # If resolution is missing (0,0), derive from the transform pixel size
        if (res_x == 0.0 and res_y == 0.0) and self.transform is not None:
            t = self.transform
            res_x = abs(t.a)
            res_y = abs(t.e)

        return GeoMetadata(
            crs=self.crs,
            bounds=self.bounds,
            band_count=self.band_count,
            width=self.width * scale,
            height=self.height * scale,
            resolution=(res_x / scale, res_y / scale),
            dtype=self.dtype,
            transform=new_transform,
            nodata=self.nodata,
            driver=self.driver,
            tags={**self.tags, "spartan_scale": str(scale)},
        )



@dataclass
class EnhancementResult:
    original_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    original_array: np.ndarray
    enhanced_array: np.ndarray
    metadata: GeoMetadata
    enhanced_metadata: GeoMetadata
    psnr: float
    ssim: float
    latency_ms: float
    model_name: str
    used_fallback: bool
    message: str = ""
    weight_source: Optional[str] = None  # "trained" | "initialised" | None



# ---------------------------------------------------------------------------
# Device / model helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_compute_device() -> Tuple[str, str]:
    """Return (device_key, human-readable label)."""
    if HAS_TORCH and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", f"CUDA GPU · {name}"
    if HAS_TORCH:
        return "cpu", "CPU (PyTorch)"
    return "cpu", "CPU (NumPy fallback)"


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device: str) -> Dict[str, Any]:
    """
    Load (or stub) a SPARTAN enhancement model.

    For SPARTAN-SwinTransformer this delegates entirely to
    ``spartan.models.registry.build_swin_model`` which:
      - Resolves weights relative to the spartan package (not cwd).
      - Auto-creates ``spartan/weights/`` if absent.
      - Loads a trained checkpoint when found, or saves random-init weights
        and returns ``source="initialised"`` so real inference always runs.
      - Logs all state-dict key / shape mismatches at ERROR level.

    SPARTAN-Lite always uses bicubic (no weights needed).
    """
    info: Dict[str, Any] = {
        "name": model_name,
        "device": device,
        "ready": False,
        "backend": "bicubic",
        "model": None,
        "weight_source": None,
        "message": "",
    }

    # ── Bicubic baseline ────────────────────────────────────────────────────
    if model_name == ModelChoice.LITE.value:
        info["ready"] = True
        info["backend"] = "bicubic"
        info["message"] = "Bicubic baseline ready (no weights required)."
        return info

    # ── SwinTransformer ─────────────────────────────────────────────────────
    if model_name == ModelChoice.SWIN.value:
        if not HAS_TORCH:
            info["ready"] = True
            info["backend"] = "bicubic"
            info["message"] = "PyTorch not installed — using bicubic fallback."
            logger.warning(info["message"])
            return info

        try:
            import sys as _sys
            # Make spartan importable regardless of launch directory.
            _spartan_parent = str(
                Path(__file__).resolve().parent.parent
            )
            if _spartan_parent not in _sys.path:
                _sys.path.insert(0, _spartan_parent)

            from spartan.models.registry import build_swin_model  # noqa: PLC0415

            model, weight_path, source = build_swin_model(device)

            info["ready"] = True
            info["backend"] = "torch"
            info["model"] = model
            info["weight_source"] = source   # "trained" | "initialised"
            info["message"] = (
                f"Successfully loaded SPARTAN-SwinTransformer weights from '{weight_path.name}'."
                if source == "trained"
                else
                f"No trained checkpoint found — running with architecture-initialized "
                f"weights saved to '{weight_path.name}'. "
                f"Real inference active; replace with a trained checkpoint "
                f"for production-quality SR."
            )
            logger.info(
                "SwinIR ready: source=%s, path=%s, device=%s",
                source, weight_path, device,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.error(
                "Failed to build SwinIR model: %s", exc, exc_info=True
            )
            raise RuntimeError(f"Failed to load SPARTAN-SwinTransformer: {exc}") from exc
        return info

    # ── SRResNet (generic torch path) ───────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    srresnet_path = project_root / "spartan" / "weights" / "spartan_srresnet.pth"

    if srresnet_path.is_file() and HAS_TORCH:
        try:
            state = torch.load(srresnet_path, map_location=device)
            info["ready"] = True
            info["backend"] = "torch"
            info["state_dict_keys"] = (
                list(state.keys())[:8] if isinstance(state, dict) else []
            )
            info["message"] = f"Loaded weights from {srresnet_path.name}."
            return info
        except Exception as exc:
            logger.error(
                "Failed to load %s: %s", srresnet_path, exc, exc_info=True
            )
            info["message"] = f"Weight load failed ({exc}); bicubic fallback."

    info["ready"] = True
    info["backend"] = "bicubic"
    if not info["message"]:
        info["message"] = (
            f"Weights for {model_name} not found — using bicubic fallback."
        )
    return info


def prepare_display_rgb(
    tensor_01: torch.Tensor | np.ndarray,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Applies joint percentile contrast stretching across RGB channels
    to preserve natural ground reflectance colors, with optional contrast gain.
    Tensor channel order: [B4 (Red), B3 (Green), B2 (Blue)].
    """
    arr = tensor_01.detach().cpu().numpy() if isinstance(tensor_01, torch.Tensor) else np.array(tensor_01)
    if arr.ndim == 4:
        arr = arr.squeeze(0)  # (3, H, W)

    # Transpose to (H, W, 3)
    rgb = np.transpose(arr[:3, :, :], (1, 2, 0))

    # Joint 2%-98% percentile stretch to preserve exact channel balance
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        stretched = (rgb - p2) / (p98 - p2)
    else:
        stretched = rgb

    stretched = np.clip(stretched, 0.0, 1.0)
    if abs(contrast - 1.0) > 1e-3:
        mid = 0.5
        stretched = np.clip((stretched - mid) * contrast + mid, 0.0, 1.0)

    return (stretched * 255.0).astype(np.uint8)


def boost_high_frequency_details(
    sr_output: torch.Tensor,
    input_resized: torch.Tensor,
    multiplier: float = 1.0,
    contrast: float = 1.0,
) -> torch.Tensor:
    """
    Adaptive high-frequency detail and edge sharpening boost on sr_output.
    Driven by the 'Enhancement multiplier' and 'Contrast' sliders.

    Combines:
      1. Super-resolution residual amplification over bicubic baseline.
      2. Spatial unsharp micro-texture enhancement to remove optical softness.
    """
    # 1. Super-resolution residual (learned high-frequency structures over bicubic)
    sr_residual = sr_output - input_resized

    # 2. Local spatial unsharp filter for fine texture and edge crispness
    device = sr_output.device
    kernel_size = 5
    sigma = 1.2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_2d = g_1d[:, None] * g_1d[None, :]
    kernel = (g_2d / g_2d.sum()).view(1, 1, kernel_size, kernel_size).repeat(sr_output.shape[1], 1, 1, 1)

    pad = kernel_size // 2
    smooth = torch.nn.functional.conv2d(sr_output, kernel, padding=pad, groups=sr_output.shape[1])
    local_edges = sr_output - smooth

    # 3. Dynamic gain:
    # Multiplier scales residual detail and unsharp edge boost.
    # Contrast modulates edge gradient punch.
    # At default multiplier=1.0, contrast=1.0, applies a balanced 0.40 boost to eliminate softness.
    sharpness_gain = 0.40 * multiplier * (0.8 + 0.2 * contrast)
    boosted = input_resized + multiplier * sr_residual + sharpness_gain * local_edges

    return torch.clamp(boosted, 0.0, 1.0)


def calculate_metrics(sr: "torch.Tensor", ref: "torch.Tensor") -> "tuple[float, float]":
    sr_c, ref_c = torch.clamp(sr, 0.0, 1.0), torch.clamp(ref, 0.0, 1.0)
    return (
        peak_signal_noise_ratio(sr_c, ref_c, data_range=1.0).item(),
        structural_similarity_index_measure(sr_c, ref_c, data_range=1.0).item(),
    )






# ---------------------------------------------------------------------------
# Raster I/O & visualization
# ---------------------------------------------------------------------------


def _synthetic_sentinel2(
    height: int = 256,
    width: int = 256,
    bands: int = 4,
) -> Tuple[np.ndarray, GeoMetadata]:
    """Generate a plausible multi-band placeholder when no file is uploaded."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:height, 0:width]
    terrain = (
        0.35 * np.sin(xx / 28.0)
        + 0.25 * np.cos(yy / 22.0)
        + 0.15 * np.sin((xx + yy) / 40.0)
    )
    terrain = (terrain - terrain.min()) / (np.ptp(terrain) + 1e-8)

    stack = []
    for b in range(bands):
        noise = rng.normal(0.0, 0.04, size=(height, width))
        channel = np.clip(terrain * (0.55 + 0.12 * b) + noise + 0.15, 0.0, 1.0)
        stack.append((channel * 10000).astype(np.float32))  # Sentinel-like DN

    array = np.stack(stack, axis=0)
    transform = (
        Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0) if HAS_RASTERIO else None
    )
    meta = GeoMetadata(
        crs="EPSG:32643",
        bounds=(500000.0, 2000000.0 - height * 10.0, 500000.0 + width * 10.0, 2000000.0),
        band_count=bands,
        width=width,
        height=height,
        resolution=(10.0, 10.0),
        dtype="float32",
        transform=transform,
        nodata=None,
        tags={"source": "synthetic_sentinel2_placeholder"},
    )
    return array, meta


@st.cache_data(show_spinner=False, ttl=3600)
def load_geotiff(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, GeoMetadata]:
    """
    Read a GeoTIFF from in-memory bytes.

    Returns (C, H, W) float32 array and geospatial metadata.
    """
    if not HAS_RASTERIO:
        logger.warning("rasterio unavailable — returning synthetic raster for %s", filename)
        return _synthetic_sentinel2()

    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                data = src.read().astype(np.float32)
                transform = src.transform
                bounds = src.bounds
                meta = GeoMetadata(
                    crs=str(src.crs) if src.crs else "Unknown",
                    bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                    band_count=src.count,
                    width=src.width,
                    height=src.height,
                    resolution=(abs(src.res[0]), abs(src.res[1])),
                    dtype=str(src.dtypes[0]),
                    transform=transform,
                    nodata=src.nodata,
                    driver=src.driver,
                    tags=dict(src.tags()),
                )
                if data.size == 0 or data.shape[0] == 0:
                    raise ValueError("GeoTIFF contains no bands.")
                return data, meta
    except Exception as exc:
        logger.exception("Corrupt or unreadable GeoTIFF (%s): %s", filename, exc)
        raise ValueError(
            f"Could not read '{filename}' as a GeoTIFF. "
            f"The file may be corrupt or unsupported. Details: {exc}"
        ) from exc


def _percentile_stretch(
    band: np.ndarray,
    lo: float = 2.0,
    hi: float = 98.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Percentile stretch a single band to [0, 1], then apply contrast gain."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)

    p_lo, p_hi = np.percentile(finite, (lo, hi))
    if p_hi <= p_lo:
        p_lo, p_hi = float(finite.min()), float(finite.max() + 1e-6)

    stretched = (band - p_lo) / (p_hi - p_lo + 1e-8)
    stretched = np.clip(stretched, 0.0, 1.0)
    # Contrast around mid-grey
    mid = 0.5
    stretched = np.clip((stretched - mid) * contrast + mid, 0.0, 1.0)
    return stretched.astype(np.float32)


def select_display_bands(
    array: np.ndarray,
    mode: BandMode,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Map a (C, H, W) stack to an (H, W, 3) RGB uint8 preview.

    Band indices assume Sentinel-2 ordering when ≥4 bands:
      0=B2(Blue), 1=B3(Green), 2=B4(Red), 3=B8(NIR)
    Falls back intelligently for fewer bands.
    """
    c, _, _ = array.shape

    def _idx(*preferred: int) -> List[int]:
        return [min(i, c - 1) for i in preferred]

    if c == 12:
        # Standard Sentinel-2 12-band product: B2=idx 1, B3=idx 2, B4=idx 3, B8=idx 7
        if mode == BandMode.RGB:
            indices = [3, 2, 1]
        elif mode == BandMode.FALSE_COLOR:
            indices = [7, 3, 2]
        elif mode == BandMode.NIR:
            indices = [7, 7, 7]
        elif mode == BandMode.RED:
            indices = [3, 3, 3]
        elif mode == BandMode.GREEN:
            indices = [2, 2, 2]
        else:  # BLUE
            indices = [1, 1, 1]
    elif mode == BandMode.RGB:
        indices = _idx(2, 1, 0) if c >= 3 else _idx(0, 0, 0)
    elif mode == BandMode.FALSE_COLOR:
        indices = _idx(3, 2, 1) if c >= 4 else _idx(0, min(1, c - 1), min(2, c - 1))
    elif mode == BandMode.NIR:
        i = min(3, c - 1)
        indices = [i, i, i]
    elif mode == BandMode.RED:
        i = min(2, c - 1)
        indices = [i, i, i]
    elif mode == BandMode.GREEN:
        i = min(1, c - 1)
        indices = [i, i, i]
    else:  # BLUE
        indices = [0, 0, 0]

    channels = [_percentile_stretch(array[i], contrast=contrast) for i in indices]
    rgb = np.stack(channels, axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def bicubic_upsample(array: np.ndarray, scale: int = TARGET_SCALE) -> np.ndarray:
    """
    Upsample (C, H, W) by `scale` using bicubic interpolation.

    Prefers rasterio (CRS-aware resampling path) then PyTorch, then NumPy zoom.
    """
    c, h, w = array.shape
    out_h, out_w = h * scale, w * scale

    if HAS_RASTERIO:
        try:
            out = np.empty((c, out_h, out_w), dtype=np.float32)
            for i in range(c):
                # rasterio.warp.reproject would need full georef; use skimage-like
                # via Affine scaling of an in-memory dataset instead.
                with MemoryFile() as memfile:
                    profile = {
                        "driver": "GTiff",
                        "height": h,
                        "width": w,
                        "count": 1,
                        "dtype": "float32",
                        "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(h)),
                    }
                    with memfile.open(**profile) as dst:
                        dst.write(array[i], 1)
                    with memfile.open() as src:
                        data = src.read(
                            1,
                            out_shape=(out_h, out_w),
                            resampling=Resampling.cubic,
                        )
                out[i] = data.astype(np.float32)
            return out
        except Exception as exc:
            logger.warning("rasterio upsample failed (%s); trying fallbacks.", exc)

    if HAS_TORCH:
        try:
            t = torch.from_numpy(array).unsqueeze(0)  # (1, C, H, W)
            up = torch.nn.functional.interpolate(
                t, scale_factor=scale, mode="bicubic", align_corners=False
            )
            return up.squeeze(0).numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("torch upsample failed (%s); using NumPy.", exc)

    # NumPy nearest-neighbour tiled upsample as last resort
    return np.repeat(np.repeat(array, scale, axis=1), scale, axis=2).astype(np.float32)


def apply_enhancement_multiplier(
    original: np.ndarray,
    enhanced: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """
    Blend residual detail: enhanced' = original_up + m * (enhanced - original_up).

    multiplier=1.0 keeps the model output; >1 exaggerates high-frequency detail.
    """
    if abs(multiplier - 1.0) < 1e-3:
        return enhanced
    # Align original to enhanced shape if needed
    if original.shape != enhanced.shape:
        scale_h = enhanced.shape[1] // original.shape[1]
        scale_w = enhanced.shape[2] // original.shape[2]
        scale = min(scale_h, scale_w) or 1
        original_up = bicubic_upsample(original, scale=scale)
        # Crop/pad to exact match
        original_up = _match_shape(original_up, enhanced.shape)
    else:
        original_up = original

    residual = enhanced - original_up
    return (original_up + multiplier * residual).astype(np.float32)


def _match_shape(arr: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    c, h, w = shape
    out = np.zeros(shape, dtype=arr.dtype)
    c_m = min(c, arr.shape[0])
    h_m = min(h, arr.shape[1])
    w_m = min(w, arr.shape[2])
    out[:c_m, :h_m, :w_m] = arr[:c_m, :h_m, :w_m]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """PSNR between two arrays of identical shape (higher is better)."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    mse = np.mean((ref - est) ** 2)
    if mse <= 1e-12:
        return 99.0
    peak = max(float(ref.max()), 1e-6)
    return float(10.0 * np.log10((peak ** 2) / mse))


def compute_ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Mean SSIM over channels (simplified luminance-only Gaussian approx).

    Avoids a hard dependency on scikit-image.
    """
    try:
        # Import optionally without requiring scikit-image for the fallback path.
        import importlib

        sk_ssim = importlib.import_module("skimage.metrics").structural_similarity

        ref = np.moveaxis(reference, 0, -1)
        est = np.moveaxis(estimate, 0, -1)
        # Match dynamic range
        data_range = float(max(ref.max() - ref.min(), 1e-6))
        channel_axis = -1 if ref.ndim == 3 else None
        score = sk_ssim(
            ref,
            est,
            data_range=data_range,
            channel_axis=channel_axis,
        )
        return float(score)
    except Exception:
        pass

    # Lightweight luminance SSIM on first (or mean) band
    ref = reference.mean(axis=0).astype(np.float64)
    est = estimate.mean(axis=0).astype(np.float64)
    data_range = max(ref.max() - ref.min(), 1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = ref.mean(), est.mean()
    sigma_x = ref.var()
    sigma_y = est.var()
    sigma_xy = ((ref - mu_x) * (est - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(np.clip(num / (den + 1e-12), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Enhancement pipeline
# ---------------------------------------------------------------------------


def run_enhancement(
    array: np.ndarray,
    metadata: GeoMetadata,
    model_info: Dict[str, Any],
    band_mode: BandMode,
    contrast: float,
    multiplier: float,
    scale: int = TARGET_SCALE,
) -> EnhancementResult:
    """Execute super-resolution (or bicubic fallback) and compute metrics."""
    t0 = time.perf_counter()
    used_fallback = model_info.get("backend") == "bicubic"
    message = model_info.get("message", "")

    # Model path — SwinIR runs a real forward pass when model_info["model"] is
    # populated.  Any exception (shape mismatch, key error, OOM, …) is logged
    # at ERROR level with a full traceback so it is visible in the console.
    if model_info.get("backend") == "torch" and HAS_TORCH:
        try:
            device = model_info.get("device", "cpu")
            raw_bands = array

            input_tensor = preprocess_array(raw_bands).to(device)

            # Inference pass
            with torch.no_grad():
                sr_output = model_info["model"](input_tensor)
                sr_output = torch.clamp(sr_output, 0.0, 1.0)

            # Bicubic-upsample input to same spatial size for metric comparison:
            _, _, out_h, out_w = sr_output.shape
            input_resized = torch.nn.functional.interpolate(
                input_tensor, size=(out_h, out_w), mode="bicubic", align_corners=False
            )
            psnr_val, ssim_val = calculate_metrics(sr_output, input_resized)

            # Adaptive high-frequency detail and sharpening boost on sr_output
            # driven by the "Enhancement multiplier" and "Contrast" sliders:
            sr_boosted = boost_high_frequency_details(
                sr_output, input_resized, multiplier=multiplier, contrast=contrast
            )

            # Display conversions with contrast control
            original_rgb = prepare_display_rgb(input_tensor, contrast=contrast)
            enhanced_rgb = prepare_display_rgb(sr_boosted, contrast=contrast)

            enhanced = sr_boosted.squeeze(0).cpu().numpy()
            used_fallback = False
            psnr = psnr_val
            ssim = ssim_val
        except Exception as exc:
            logger.error(
                "Model inference failed for '%s': %s",
                model_info.get("name"),
                exc,
                exc_info=True,
            )
            enhanced = bicubic_upsample(array, scale=scale)
            used_fallback = True
            message = f"Inference error ({exc}); bicubic fallback applied."
            reference = bicubic_upsample(array, scale=scale)
            reference = _match_shape(reference, enhanced.shape)
            psnr = compute_psnr(reference, enhanced)
            ssim = compute_ssim(reference, enhanced)
            original_rgb = select_display_bands(array, band_mode, contrast=contrast)
            enhanced_rgb = select_display_bands(enhanced, band_mode, contrast=contrast)
    else:
        # Bicubic baseline: arrays remain in raw DN-scale throughout.
        enhanced = bicubic_upsample(array, scale=scale)
        enhanced = apply_enhancement_multiplier(array, enhanced, multiplier)
        reference = bicubic_upsample(array, scale=scale)
        reference = _match_shape(reference, enhanced.shape)
        psnr = compute_psnr(reference, enhanced)
        ssim = compute_ssim(reference, enhanced)
        original_rgb = select_display_bands(array, band_mode, contrast=contrast)
        enhanced_rgb = select_display_bands(enhanced, band_mode, contrast=contrast)

    enhanced_meta = metadata.upsampled(scale)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return EnhancementResult(
        original_rgb=original_rgb,
        enhanced_rgb=enhanced_rgb,
        original_array=array,
        enhanced_array=enhanced,
        metadata=metadata,
        enhanced_metadata=enhanced_meta,
        psnr=psnr,
        ssim=ssim,
        latency_ms=latency_ms,
        model_name=model_info.get("name", "unknown"),
        used_fallback=used_fallback,
        message=message,
        weight_source=model_info.get("weight_source"),  # "trained"|"initialised"|None
    )


def _torch_model_infer(
    array: np.ndarray,
    model_info: Dict[str, Any],
    scale: int,
) -> np.ndarray:
    """
    Run a real SwinIR (or other registered) torch forward pass.

    Expects model_info["model"] to be a fully initialised, weight-loaded
    nn.Module placed on model_info["device"].  Raises NotImplementedError for
    model types that are not yet wired so the caller's except clause can
    choose a fallback — but those errors will be logged at ERROR level, not
    silently swallowed.
    """
    model = model_info.get("model")
    if model is None:
        raise NotImplementedError(
            f"No nn.Module found in model_info for '{model_info.get('name')}'. "
            "The model was not loaded successfully."
        )

    device = model_info.get("device", "cpu")
    input_tensor = preprocess_array(array).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    out_np = output_tensor.squeeze(0).cpu().numpy()
    return out_np.astype(np.float32)


def geotiff_to_bytes(
    array: np.ndarray,
    metadata: GeoMetadata,
) -> bytes:
    """Serialize (C, H, W) array to an in-memory GeoTIFF preserving CRS."""
    if not HAS_RASTERIO:
        # Fallback: raw .npy so the download button still works in degraded mode
        buf = io.BytesIO()
        np.save(buf, array)
        return buf.getvalue()

    c, h, w = array.shape
    transform = metadata.transform
    if transform is None:
        res_x, res_y = metadata.resolution
        transform = Affine(res_x, 0.0, metadata.bounds[0], 0.0, -res_y, metadata.bounds[3])

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "float32",
        "crs": metadata.crs if metadata.crs != "Unknown" else None,
        "transform": transform,
        "compress": "lzw",
        "nodata": metadata.nodata,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(array.astype(np.float32))
            if metadata.tags:
                dst.update_tags(**{k: str(v) for k, v in metadata.tags.items()})
            dst.update_tags(
                SPARTAN_VERSION=APP_VERSION,
                SPARTAN_NOTE="Super-resolved by SPARTAN",
            )
        return memfile.read()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        [data-testid="stAppDeployButton"] {display:none;}
        [data-testid="stToolbarActions"] {display:none;}
        [data-testid="stToolbar"] {visibility: hidden; display: none;}
        [data-testid="stHeader"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --spartan-ink: #0b1f2a;
            --spartan-teal: #1a7a6d;
            --spartan-sand: #e8efe9;
            --spartan-amber: #2f5a9e;
            --spartan-slate: #3d5563;
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .spartan-hero {
            background: linear-gradient(135deg, #0b1f2a 0%, #1a4a52 55%, #1a7a6d 100%);
            color: #f4faf7;
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .spartan-hero h1 {
            font-size: 1.75rem;
            margin: 0 0 0.25rem 0;
            letter-spacing: 0.04em;
            font-weight: 700;
        }
        .spartan-hero p {
            margin: 0;
            opacity: 0.9;
            font-size: 0.95rem;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
        }
        .badge.ok { background: rgba(26,122,109,0.45); }
        .badge.warn { background: rgba(196,122,34,0.45); }

        .metric-card {
            background: var(--spartan-sand);
            border: 1px solid #c9d6ce;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            text-align: center;
        }
        .metric-card .label {
            font-size: 0.75rem;
            color: var(--spartan-slate);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .metric-card .value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.45rem;
            font-weight: 500;
            color: var(--spartan-ink);
            margin-top: 0.2rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1f2a 0%, #12323a 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #e8efe9 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stFileUploader label {
            color: #c9d6ce !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(device_label: str, model_state: str, model_ok: bool) -> None:
    st.markdown(
        """
        <div class="spartan-hero">
            <h1>SPARTAN</h1>
            <p>Satellite Pixel-Augmented Resolution &amp; Terrain Analysis Network</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data acquisition pipeline helpers
# ---------------------------------------------------------------------------


# Project root — one level above the app/ directory
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_RAW_TIF = Path(_PROJECT_ROOT) / "spartan" / "data" / "raw" / "aoi_scene.tif"
_CLEAN_TIF = Path(_PROJECT_ROOT) / "spartan" / "data" / "processed" / "clean_scene.tif"
_PATCHES_DIR = Path(_PROJECT_ROOT) / "spartan" / "data" / "processed" / "patches"


@dataclass
class FetchParams:
    """Parameters the user enters for data acquisition."""
    lat: float
    lon: float
    start_date: str
    end_date: str
    patch_size: int


def run_fetch_pipeline(params: FetchParams) -> str:
    """
    Execute the Module 1 pipeline (fetch → preprocess → tile) in a subprocess.

    Returns the path to the clean GeoTIFF on success, raises on failure.
    """
    cmd = [
        sys.executable, "train.py",
        "--lat", str(params.lat),
        "--lon", str(params.lon),
        "--start", params.start_date,
        "--end", params.end_date,
        "--patch-size", str(params.patch_size),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    if not _CLEAN_TIF.is_file():
        raise FileNotFoundError(
            f"Pipeline finished but output not found at {_CLEAN_TIF}"
        )
    return str(_CLEAN_TIF)


# ---------------------------------------------------------------------------
# Sidebar — supports two input modes
# ---------------------------------------------------------------------------


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### SPARTAN")
        st.markdown("---")

        # ---- Input mode selector ----
        input_mode = st.radio(
            "📥 Input source",
            options=["Upload GeoTIFF", "Fetch from Sentinel-2"],
            index=0,
            help="Upload an existing file, or fetch new data by coordinates and date range.",
            horizontal=True,
        )

        # ---- Mode 1: File upload (original behaviour) ----
        uploaded = None
        fetch_clicked = False
        fetch_params: Optional[FetchParams] = None

        if input_mode == "Upload GeoTIFF":
            uploaded = st.file_uploader(
                "Upload Sentinel-2 GeoTIFF",
                type=["tif", "tiff"],
                accept_multiple_files=False,
                help="Single- or multi-band GeoTIFF at ~10 m resolution.",
            )
        else:
            # ---- Mode 2: Live data fetch ----
            st.markdown("#### 🛰️ Area of Interest")
            col_lat, col_lon = st.columns(2)
            with col_lat:
                lat = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=19.0760,
                    step=0.0001,
                    format="%.4f",
                    help="Decimal degrees (e.g. 19.0760 for Mumbai).",
                )
            with col_lon:
                lon = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=72.8777,
                    step=0.0001,
                    format="%.4f",
                    help="Decimal degrees (e.g. 72.8777 for Mumbai).",
                )

            st.markdown("#### 📅 Date Range")
            col_start, col_end = st.columns(2)
            with col_start:
                start = st.date_input(
                    "Start date",
                    value=date(2025, 11, 1),
                    help="First day of the observation window.",
                )
            with col_end:
                end = st.date_input(
                    "End date",
                    value=date(2026, 3, 1),
                    help="Last day of the observation window.",
                )

            st.markdown("#### 🧩 Patch Settings")
            patch_size = st.select_slider(
                "Patch dimensions (px)",
                options=[64, 128, 256, 512],
                value=256,
                help="Pixel size of each tile the clean raster is split into.",
            )

            # Quick validation
            if start > end:
                st.warning("⚠️ Start date is after end date — please adjust.")

            fetch_clicked = st.button(
                "🚀 Fetch & Process",
                type="primary",
                use_container_width=True,
                disabled=(start > end),
            )

            fetch_params = FetchParams(
                lat=lat,
                lon=lon,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                patch_size=patch_size,
            )

        st.markdown("---")

        # ---- Common enhancement controls (visible in both modes) ----
        st.markdown("#### 🔧 Enhancement")
        band_mode = st.selectbox(
            "Band composition",
            options=[m.value for m in BandMode],
            index=0,
        )

        model_name = st.selectbox(
            "Enhancement model",
            options=[m.value for m in ModelChoice],
            index=0,
        )

        contrast = st.slider("Contrast", min_value=0.5, max_value=2.5, value=1.0, step=0.05)
        multiplier = st.slider(
            "Enhancement multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="Scales high-frequency residual relative to bicubic upsample.",
        )
        run = st.button("Run Enhancement", type="primary", use_container_width=True)

    return {
        "uploaded": uploaded,
        "band_mode": BandMode(band_mode),
        "model_name": model_name,
        "contrast": float(contrast),
        "multiplier": float(multiplier),
        "run": bool(run),
        "input_mode": input_mode,
        "fetch_clicked": fetch_clicked,
        "fetch_params": fetch_params,
    }


def render_comparison(original_rgb: np.ndarray, enhanced_rgb: np.ndarray) -> None:
    st.subheader("Interactive visual comparison")
    st.caption("Original (10 m) vs SPARTAN Enhanced (<2.5 m)")

    if HAS_IMAGE_COMPARISON and _image_comparison is not None:
        try:
            _image_comparison(
                img1=original_rgb,
                img2=enhanced_rgb,
                label1="Original (10 m)",
                label2="SPARTAN Enhanced (<2.5 m)",
                width=700,
            )
            return
        except Exception as exc:
            logger.warning("image_comparison failed (%s); using tabs.", exc)

    tab_side, tab_orig, tab_enh = st.tabs(["Side-by-side", "Original", "Enhanced"])
    with tab_side:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original (10 m)**")
            st.image(original_rgb, use_container_width=True, clamp=True)
        with col_b:
            st.markdown("**SPARTAN Enhanced (<2.5 m)**")
            st.image(enhanced_rgb, use_container_width=True, clamp=True)
    with tab_orig:
        st.image(original_rgb, use_container_width=True, clamp=True, caption="Original (10 m)")
    with tab_enh:
        st.image(
            enhanced_rgb,
            use_container_width=True,
            clamp=True,
            caption="SPARTAN Enhanced (<2.5 m)",
        )


def render_metadata(meta: GeoMetadata, enhanced_meta: GeoMetadata) -> None:
    with st.expander("Geospatial metadata inspector", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Source**")
            st.write(
                {
                    "CRS": meta.crs,
                    "Bounds (L, B, R, T)": meta.bounds,
                    "Band count": meta.band_count,
                    "Size (W×H)": f"{meta.width} × {meta.height}",
                    "Resolution": tuple(
                        round(r, 6) if r >= 0.001 else float(f"{r:.6g}")
                        for r in meta.resolution
                    ),
                    "Dtype": meta.dtype,
                    "NoData": meta.nodata,
                    "Driver": meta.driver,
                }
            )
        with c2:
            st.markdown("**After SPARTAN upsample**")
            st.write(
                {
                    "CRS": enhanced_meta.crs,
                    "Bounds (L, B, R, T)": enhanced_meta.bounds,
                    "Band count": enhanced_meta.band_count,
                    "Size (W×H)": f"{enhanced_meta.width} × {enhanced_meta.height}",
                    "Resolution": tuple(
                        round(r, 6) if r >= 0.001 else float(f"{r:.6g}")
                        for r in enhanced_meta.resolution
                    ),
                    "Scale factor": TARGET_SCALE,
                    "Tags": enhanced_meta.tags,
                }
            )


def render_metrics(result: EnhancementResult) -> None:
    st.subheader("Analytics & metrics")
    cols = st.columns(3)
    cards = [
        ("PSNR", f"{result.psnr:.2f} dB"),
        ("SSIM", f"{result.ssim:.4f}"),
        ("Latency", f"{result.latency_ms:.1f} ms"),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result.used_fallback:
        st.warning(result.message or "Running in bicubic fallback mode.")
    else:
        st.success(f"Successfully loaded {result.model_name} weights")
        if result.message and result.message != f"Successfully loaded {result.model_name} weights":
            if result.weight_source == "initialised":
                st.info(result.message)
            else:
                st.caption(result.message)


def render_export(result: EnhancementResult, source_name: str) -> None:
    st.subheader("Export")
    stem = Path(source_name).stem or "spartan"
    out_name = f"{stem}_spartan_sr.tif" if HAS_RASTERIO else f"{stem}_spartan_sr.npy"

    try:
        payload = geotiff_to_bytes(result.enhanced_array, result.enhanced_metadata)
        mime = "image/tiff" if HAS_RASTERIO else "application/octet-stream"
        st.download_button(
            label="Download Super-Resolved GeoTIFF",
            data=payload,
            file_name=out_name,
            mime=mime,
            type="primary",
            use_container_width=False,
        )
        if not HAS_RASTERIO:
            st.warning("rasterio missing — export is NumPy `.npy`, not GeoTIFF.")
    except Exception as exc:
        st.error(f"Failed to build download payload: {exc}")


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="SPARTAN · Geospatial Super-Resolution",
        page_icon="🛰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    device_key, device_label = get_compute_device()
    controls = render_sidebar()

    model_info = load_model(controls["model_name"], device_key)
    model_state = (
        f"{model_info['name']} · {model_info['backend']}"
        if model_info.get("ready")
        else "Model unavailable"
    )
    render_header(device_label, model_state, bool(model_info.get("ready")))
    if controls["model_name"] == ModelChoice.SWIN.value:
        if model_info.get("backend") == "torch":
            weight_source = model_info.get("weight_source")
            if weight_source == "trained":
                st.success(
                    f"✅ Successfully loaded trained SPARTAN-SwinTransformer weights — "
                    f"{model_info.get('message', '')}"
                )
            elif weight_source == "initialised":
                st.warning(
                    "⚠️ Running with un-trained initialized weights. "
                    "No trained checkpoint was found in `spartan/weights/`. "
                    "Inference is active but output quality will be low — "
                    "replace with a trained `.pth` checkpoint for production SR."
                )
            else:
                st.info(model_info.get("message", "SPARTAN-SwinTransformer ready."))
        else:
            st.warning(model_info.get("message", "Model running with fallback."))

    # -----------------------------------------------------------------
    # Handle "Fetch from Sentinel-2" mode
    # -----------------------------------------------------------------
    if controls["input_mode"] == "Fetch from Sentinel-2" and controls["fetch_clicked"]:
        params = controls["fetch_params"]
        if params is not None:
            st.markdown("---")
            st.subheader("📡 Data Acquisition Pipeline")
            st.caption(
                f"Lat {params.lat:.4f} · Lon {params.lon:.4f} · "
                f"{params.start_date} → {params.end_date} · "
                f"Patch {params.patch_size} px"
            )

            progress = st.progress(0, text="Initialising pipeline…")
            status_box = st.empty()

            try:
                # Step indicators
                progress.progress(10, text="🛰️  Step 1/3 — Fetching Sentinel-2 data…")
                status_box.info("Connecting to Copernicus STAC API and downloading scene…")
                tif_path = run_fetch_pipeline(params)
                progress.progress(75, text="✅ Data fetched & preprocessed successfully!")
                status_box.success(
                    f"Clean GeoTIFF saved to `{tif_path}`\n\n"
                    f"Patches written to `{_PATCHES_DIR}`"
                )
                progress.progress(100, text="Pipeline complete ✔")

                # Store path so downstream can load it
                st.session_state["fetched_tif_path"] = tif_path
            except Exception as exc:
                progress.progress(0, text="❌ Pipeline failed")
                status_box.error(f"Pipeline error: {exc}")
                st.stop()

    # -----------------------------------------------------------------
    # Resolve input raster (upload, fetched, or synthetic fallback)
    # -----------------------------------------------------------------
    source_name = "synthetic_sentinel2.tif"
    array: Optional[np.ndarray] = None
    metadata: Optional[GeoMetadata] = None
    load_error: Optional[str] = None

    uploaded = controls["uploaded"]
    fetched_path: Optional[str] = st.session_state.get("fetched_tif_path")

    if uploaded is not None:
        # User uploaded a file via drag-and-drop
        source_name = uploaded.name
        suffix = Path(source_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            load_error = f"Unsupported extension '{suffix}'. Use .tif / .tiff."
        else:
            try:
                array, metadata = load_geotiff(uploaded.getvalue(), source_name)
            except ValueError as exc:
                load_error = str(exc)

    elif fetched_path and Path(fetched_path).is_file():
        # Data was just fetched via the pipeline
        source_name = Path(fetched_path).name
        try:
            raw_bytes = Path(fetched_path).read_bytes()
            array, metadata = load_geotiff(raw_bytes, source_name)
        except (ValueError, Exception) as exc:
            load_error = f"Could not read fetched file: {exc}"

    if load_error:
        st.error(load_error)
        st.stop()

    is_synthetic = False
    if array is None or metadata is None:
        array, metadata = _synthetic_sentinel2()
        source_name = "synthetic_sentinel2.tif"
        is_synthetic = True
    elif metadata.tags.get("source") == "synthetic_sentinel2_placeholder":
        is_synthetic = True

    # Preview before run
    if is_synthetic:
        st.info("Upload a Sentinel-2 GeoTIFF (.tif) in the sidebar to begin enhancement.")
    else:
        preview = select_display_bands(
            array, controls["band_mode"], contrast=controls["contrast"]
        )
        with st.expander("Input preview", expanded=not controls["run"]):
            st.image(preview, caption=f"Preview · {source_name}", use_container_width=True)
            res_val = metadata.resolution[0]
            res_display = f"{res_val:.2f} m/px" if res_val >= 0.1 else f"{res_val:.6g} deg/px"
            st.caption(
                f"{metadata.band_count} bands · {metadata.width}×{metadata.height} px · "
                f"{res_display} · {metadata.crs}"
            )

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if controls["run"]:
        with st.spinner("Running SPARTAN enhancement…"):
            try:
                result = run_enhancement(
                    array=array,
                    metadata=metadata,
                    model_info=model_info,
                    band_mode=controls["band_mode"],
                    contrast=controls["contrast"],
                    multiplier=controls["multiplier"],
                    scale=TARGET_SCALE,
                )
                st.session_state["last_result"] = result
                st.session_state["last_source"] = source_name
            except Exception as exc:
                logger.exception("Enhancement failed")
                st.error(f"Enhancement failed: {exc}")
                st.stop()

    result: Optional[EnhancementResult] = st.session_state.get("last_result")
    if result is None:
        st.info("Configure options in the sidebar, then click **Run Enhancement**.")
        return

    render_comparison(result.original_rgb, result.enhanced_rgb)
    render_metadata(result.metadata, result.enhanced_metadata)
    render_metrics(result)
    render_export(result, st.session_state.get("last_source", source_name))


if __name__ == "__main__":
    main()
