from typing import Any, Dict
import numpy as np
from PIL import Image

# Pillow resampling compatibility (silences PyCharm/stubs issues)
try:
    _BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 9
except AttributeError:
    _BILINEAR = Image.BILINEAR  # older Pillow

def _downsample_for_metrics(img: Image.Image, max_side: int = 512) -> Image.Image:
    if not isinstance(max_side, int) or max_side < 32:
        raise ValueError(f"max_side must be int >= 32, got {max_side!r}")

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {img.size}")

    if max(w, h) <= max_side:
        return img

    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=_BILINEAR)

def _rgb_to_hsv_saturation_mean(rgb: np.ndarray) -> float:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB array HxWx3, got shape={rgb.shape}")

    rgb_f = rgb.astype(np.float32) / 255.0
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    s = np.zeros_like(cmax, dtype=np.float32)
    mask = cmax > 1e-8
    s[mask] = delta[mask] / cmax[mask]

    sat_mean = float(np.mean(s))
    if not np.isfinite(sat_mean):
        raise ValueError("sat_mean is non-finite")
    return sat_mean

def _colorfulness_hasler_susstrunk(rgb: np.ndarray) -> float:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB array HxWx3, got shape={rgb.shape}")

    rgb_f = rgb.astype(np.float32)
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]

    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))

    if not all(np.isfinite([std_rg, std_yb, mean_rg, mean_yb])):
        raise ValueError("colorfulness components are non-finite")

    sigma = (std_rg ** 2 + std_yb ** 2) ** 0.5
    mu = (mean_rg ** 2 + mean_yb ** 2) ** 0.5
    cf = sigma + 0.3 * mu

    if not np.isfinite(cf):
        raise ValueError("colorfulness is non-finite")
    return float(cf)

def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D array, got shape={gray.shape}")
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        raise ValueError(f"Image too small for Laplacian: shape={gray.shape}")

    g = gray.astype(np.float32)
    center = g[1:-1, 1:-1]
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * center)

    var = float(np.var(lap))
    if not np.isfinite(var):
        raise ValueError("blur_laplacian_var is non-finite")
    return var

def _bucketize(
    *,
    luma_mean: float,
    dark_frac: float,
    bright_frac: float,
    blur_laplacian_var: float,
    contrast_luma_std: float,
    sat_mean: float,
    colorfulness: float,
    is_grayscale: bool,
) -> Dict[str, str]:
    # Lighting buckets (luma in 0..255)
    if luma_mean < 45.0 or dark_frac > 0.60:
        lighting_bucket = "night"
    elif bright_frac > 0.08 and luma_mean > 170.0:
        lighting_bucket = "glare"
    elif luma_mean > 175.0:
        lighting_bucket = "bright"
    elif luma_mean < 85.0 or dark_frac > 0.35:
        lighting_bucket = "low_light"
    else:
        lighting_bucket = "normal"

    # Blur buckets (depends on downsample and laplacian implementation)
    if blur_laplacian_var >= 250.0:
        blur_bucket = "sharp"
    elif blur_laplacian_var >= 80.0:
        blur_bucket = "mild_blur"
    else:
        blur_bucket = "blurry"

    # Contrast buckets (stddev in 0..255)
    if contrast_luma_std < 35.0:
        contrast_bucket = "low"
    elif contrast_luma_std < 65.0:
        contrast_bucket = "medium"
    else:
        contrast_bucket = "high"

    # Color buckets
    # For grayscale images, color metrics are not meaningful -> force "low"
    if is_grayscale:
        color_bucket = "low"
    else:
        if sat_mean < 0.20 or colorfulness < 15.0:
            color_bucket = "low"
        elif sat_mean < 0.40 or colorfulness < 30.0:
            color_bucket = "medium"
        else:
            color_bucket = "high"

    return {
        "lighting_bucket": lighting_bucket,
        "blur_bucket": blur_bucket,
        "contrast_bucket": contrast_bucket,
        "color_bucket": color_bucket,
    }

def compute_image_quality_features(img: Image.Image, *, max_side: int = 512) -> Dict[str, Any]:
    """
    Compute deterministic image-quality / condition features.

    For grayscale images (1 band), color metrics are set to:
      sat_mean = 0.0
      colorfulness = 0.0
      color_bucket = "low"

    Raises only on computation/format errors.
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"img must be PIL.Image.Image, got {type(img).__name__}")

    bands = len(img.getbands())
    if bands not in (1, 3):
        raise ValueError(f"Unsupported band count for quality metrics: {bands}, expected 1 or 3")

    is_grayscale = (bands == 1)

    img2 = _downsample_for_metrics(img, max_side=max_side)

    # Luma/grayscale in 0..255
    gray = np.asarray(img2.convert("L"), dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError(f"Unexpected grayscale array shape: {gray.shape}")

    flat = gray.reshape(-1)
    if flat.size == 0:
        raise ValueError("Empty image array after conversion")

    luma_mean = float(np.mean(flat))
    luma_p10 = float(np.percentile(flat, 10.0))
    luma_p90 = float(np.percentile(flat, 90.0))
    if not all(np.isfinite([luma_mean, luma_p10, luma_p90])):
        raise ValueError("Non-finite luma stats")

    # Fractions
    dark_frac = float(np.mean(flat < 30.0))
    bright_frac = float(np.mean(flat > 225.0))
    if not all(np.isfinite([dark_frac, bright_frac])):
        raise ValueError("Non-finite dark/bright fractions")

    # Contrast
    contrast_luma_std = float(np.std(flat))
    contrast_luma_p90_p10 = float(luma_p90 - luma_p10)
    if not all(np.isfinite([contrast_luma_std, contrast_luma_p90_p10])):
        raise ValueError("Non-finite contrast stats")

    # Blur
    blur_laplacian_var = _laplacian_variance(gray)

    # Color metrics
    if is_grayscale:
        sat_mean = 0.0
        colorfulness = 0.0
    else:
        rgb = np.asarray(img2.convert("RGB"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Unexpected RGB array shape: {rgb.shape}")
        sat_mean = _rgb_to_hsv_saturation_mean(rgb)
        colorfulness = _colorfulness_hasler_susstrunk(rgb)

    buckets = _bucketize(
        luma_mean=luma_mean,
        dark_frac=dark_frac,
        bright_frac=bright_frac,
        blur_laplacian_var=blur_laplacian_var,
        contrast_luma_std=contrast_luma_std,
        sat_mean=sat_mean,
        colorfulness=colorfulness,
        is_grayscale=is_grayscale,
    )

    return {
        "luma_mean": luma_mean,
        "luma_p10": luma_p10,
        "luma_p90": luma_p90,
        "dark_frac": dark_frac,
        "bright_frac": bright_frac,
        "contrast_luma_std": contrast_luma_std,
        "contrast_luma_p90_p10": contrast_luma_p90_p10,
        "blur_laplacian_var": blur_laplacian_var,
        "sat_mean": float(sat_mean),
        "colorfulness": float(colorfulness),
        **buckets,
    }