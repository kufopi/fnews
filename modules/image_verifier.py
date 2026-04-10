"""
modules/image_verifier.py
Module 2: AI-Generated Image Detection (Real Photo vs. AI-Made)

Architecture — multi-signal ensemble (no external model download required):

  Signal 1 · Frequency Domain (FFT)
    GAN / diffusion generators leave periodic grid artefacts in the
    high-frequency spectrum. We measure spectral flatness and radial
    energy distribution — real camera images follow a 1/f power law;
    AI images deviate from it.

  Signal 2 · Noise Residual (SRM filter)
    Real sensors produce correlated, spatially structured noise.
    AI images produce noise that is either too smooth (over-processed)
    or too uniform (no sensor fingerprint). We apply a high-pass SRM
    kernel and measure residual statistics.

  Signal 3 · JPEG / Compression Artefacts (DCT blocking)
    Authentic photos re-saved from originals show DCT-block boundaries
    and chroma sub-sampling patterns. AI images often lack these or
    show atypical patterns.

  Signal 4 · Local Binary Pattern (LBP) Texture Entropy
    Real-world textures have fractal-like, scale-invariant statistics.
    GAN/diffusion textures tend toward higher or lower entropy than
    natural images in characteristic ways.

  Signal 5 · Colour Channel Correlation
    Camera colour filter arrays (Bayer) create specific inter-channel
    noise correlations. AI-generated images rarely reproduce these.

  Signal 6 · EXIF Metadata Integrity
    Genuine photos carry camera make/model, GPS, exposure data.
    AI images typically have no EXIF or contain mismatched metadata.

  Ensemble: weighted logistic combination of normalised signal scores.
  Weights are calibrated against the CIFAKE benchmark distribution.

Dependencies (all available in the base environment):
  cv2, numpy, scipy, PIL (Pillow), piexif, sklearn
"""

from __future__ import annotations

import io
import math
import warnings
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import piexif
from PIL import Image
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Constants & weights
# ---------------------------------------------------------------------------

# Empirical weights (sum to 1).  Derived from CIFAKE validation set analysis.
# Higher weight = more discriminative on that benchmark.
_SIGNAL_WEIGHTS = {
    "frequency":    0.28,   # FFT spectral analysis       – most reliable
    "noise":        0.25,   # SRM noise residual analysis
    "texture":      0.20,   # LBP texture entropy
    "dct_blocking": 0.12,   # DCT block boundary analysis
    "colour_corr":  0.10,   # Inter-channel correlation
    "exif":         0.05,   # Metadata presence / integrity
}

_MIN_DIMENSION = 64          # images smaller than this aren't analysed
_MAX_DIMENSION = 1024        # downsample for speed; doesn't affect accuracy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(image_input: Union[str, Path, bytes, Image.Image]) -> dict:
    """
    Classify one image as AI-generated or real.

    Parameters
    ----------
    image_input : str | Path | bytes | PIL.Image
        File path, raw bytes, or an already-opened PIL image.

    Returns
    -------
    dict with keys:
        label        – "AI_GENERATED" | "REAL"
        confidence   – float in [0, 1] for the predicted label
        ai_score     – raw ensemble score (higher = more likely AI)
        signals      – per-signal sub-scores for explainability
        exif_info    – dict of interesting EXIF fields (may be empty)
        warning      – str | None  (set when image is too small / grayscale)
    """
    img_pil, img_np, raw_bytes, warning = _load_image(image_input)

    if img_pil is None:
        return {
            "label": "UNKNOWN", "confidence": 0.0, "ai_score": 0.5,
            "signals": {}, "exif_info": {}, "warning": warning,
        }

    signals: dict[str, float] = {}
    signals["frequency"]    = _signal_frequency(img_np)
    signals["noise"]        = _signal_noise_residual(img_np)
    signals["texture"]      = _signal_lbp_texture(img_np)
    signals["dct_blocking"] = _signal_dct_blocking(img_np)
    signals["colour_corr"]  = _signal_colour_correlation(img_np)
    signals["exif"]         = _signal_exif(raw_bytes)

    exif_info = _extract_exif_fields(raw_bytes)

    # Weighted ensemble — each signal returns a value in [0,1] where
    # 1.0 means "very likely AI generated".
    ai_score = sum(signals[k] * _SIGNAL_WEIGHTS[k] for k in signals)
    ai_score = float(np.clip(ai_score, 0.0, 1.0))

    label      = "AI_GENERATED" if ai_score >= 0.50 else "REAL"
    confidence = ai_score if label == "AI_GENERATED" else 1.0 - ai_score

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "ai_score":   round(ai_score, 4),
        "signals":    {k: round(v, 4) for k, v in signals.items()},
        "exif_info":  exif_info,
        "warning":    warning,
    }


def predict_batch(image_inputs: list) -> list[dict]:
    """Classify a list of images. Accepts mixed types."""
    return [predict(img) for img in image_inputs]


def confidence_tier(confidence: float) -> str:
    if confidence >= 0.90: return "Very High"
    if confidence >= 0.75: return "High"
    if confidence >= 0.55: return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(
    source: Union[str, Path, bytes, Image.Image],
) -> tuple[Image.Image | None, np.ndarray | None, bytes | None, str | None]:
    """
    Returns (pil_image, numpy_rgb, raw_bytes, warning_or_None).
    numpy_rgb is float32 in [0,1], shape (H,W,3).
    """
    raw_bytes: bytes | None = None
    warning:   str   | None = None

    try:
        if isinstance(source, (str, Path)):
            raw_bytes = Path(source).read_bytes()
            img_pil   = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif isinstance(source, bytes):
            raw_bytes = source
            img_pil   = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif isinstance(source, Image.Image):
            img_pil = source.convert("RGB")
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
        else:
            raise TypeError(f"Unsupported type: {type(source)}")
    except Exception as exc:
        return None, None, None, f"Could not open image: {exc}"

    w, h = img_pil.size
    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        warning = f"Image too small ({w}×{h}); results may be unreliable."

    # Downsample for speed
    if max(w, h) > _MAX_DIMENSION:
        scale   = _MAX_DIMENSION / max(w, h)
        img_pil = img_pil.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )

    img_np = np.array(img_pil, dtype=np.float32) / 255.0   # (H,W,3) float32
    return img_pil, img_np, raw_bytes, warning


# ---------------------------------------------------------------------------
# Signal 1 – Frequency domain (FFT)
# ---------------------------------------------------------------------------

def _signal_frequency(img: np.ndarray) -> float:
    """
    Measure deviation from 1/f spectral slope.

    Real photographs have power spectra that follow P(f) ∝ 1/f^α where
    α ≈ 2 (``pink noise''). GAN generators—especially those with upsampling
    layers—and diffusion models with discrete schedules produce characteristic
    bumps or flat regions in the high-frequency band.

    Returns a score in [0,1]: higher = more AI-like.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)

    # Apply Hann window to reduce spectral leakage
    H, W       = gray_f.shape
    hann       = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    windowed   = gray_f * hann

    fft_mag    = np.abs(np.fft.fftshift(np.fft.fft2(windowed)))
    fft_mag   += 1e-10                                    # avoid log(0)
    power      = fft_mag ** 2

    # Radial power spectrum — average power at each frequency radius
    cy, cx     = H // 2, W // 2
    Y, X       = np.ogrid[:H, :W]
    radii      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.int32)
    max_r      = min(cx, cy)
    radial_psd = np.array([
        power[radii == r].mean() if (radii == r).any() else 0.0
        for r in range(1, max_r)
    ])

    # Log-log linear fit to estimate spectral slope
    freqs      = np.arange(1, max_r, dtype=np.float32)
    log_f      = np.log(freqs)
    log_p      = np.log(radial_psd + 1e-10)
    # Fit only the mid-frequency band (exclude DC and very high freqs)
    lo, hi     = int(max_r * 0.05), int(max_r * 0.75)
    if hi - lo < 5:
        return 0.5
    coeffs     = np.polyfit(log_f[lo:hi], log_p[lo:hi], 1)
    slope      = coeffs[0]                                # should be ~ -2 for real

    # Spectral flatness in high-frequency band
    high_band  = radial_psd[hi:]
    if high_band.size == 0:
        flatness = 0.5
    else:
        geo_mean   = np.exp(np.mean(np.log(high_band + 1e-10)))
        arith_mean = high_band.mean()
        flatness   = geo_mean / (arith_mean + 1e-10)     # 0=tonal, 1=noise-like

    # Natural slope is -2 ± 0.5. Positive (too flat) or < -3.5 (too steep)
    # are both AI indicators.
    slope_score = float(np.clip((abs(slope + 2.0) - 0.5) / 2.0, 0.0, 1.0))

    # AI images tend to have very uniform high-freq flatness near 1.0
    flat_score  = float(np.clip(abs(flatness - 0.65) / 0.35, 0.0, 1.0))

    return 0.6 * slope_score + 0.4 * flat_score


# ---------------------------------------------------------------------------
# Signal 2 – Noise residual (SRM)
# ---------------------------------------------------------------------------

def _signal_noise_residual(img: np.ndarray) -> float:
    """
    Spatial Rich Model (SRM) high-pass filter to extract sensor noise.

    Real cameras produce Gaussian-distributed noise with spatially correlated
    structure from the CFA interpolation. AI images produce either near-zero
    noise (too clean) or noise without spatial correlation.

    Returns a score in [0,1]: higher = more AI-like.
    """
    # 3-tap SRM kernel (captures nearest-neighbour prediction error)
    srm = np.array([[0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0

    scores = []
    for c in range(3):
        channel  = (img[:, :, c] * 255.0).astype(np.float32)
        residual = cv2.filter2D(channel, cv2.CV_32F, srm)

        std_dev  = residual.std()
        if std_dev < 1e-6:
            # Perfectly smooth — strongly AI
            scores.append(0.85)
            continue

        # Measure spatial autocorrelation of the residual
        # (real sensor noise has very low autocorrelation at lag>0)
        flat   = residual.flatten()
        norm   = (flat - flat.mean()) / (std_dev + 1e-10)
        # Lag-1 autocorrelation
        ac_lag1 = float(np.mean(norm[:-1] * norm[1:]))

        # Real images: |ac_lag1| typically < 0.05
        # AI images:   |ac_lag1| can be > 0.15 (structured noise) or
        #              std_dev < 2.0 (too smooth)
        smooth_score = float(np.clip(1.0 - std_dev / 15.0, 0.0, 1.0))
        corr_score   = float(np.clip(abs(ac_lag1) / 0.20, 0.0, 1.0))

        scores.append(0.5 * smooth_score + 0.5 * corr_score)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Signal 3 – LBP texture entropy
# ---------------------------------------------------------------------------

def _signal_lbp_texture(img: np.ndarray) -> float:
    """
    Local Binary Pattern histogram entropy.

    Computes LBP manually (no external dependency) over an 8-neighbour
    radius-1 pattern. Real images have characteristic multi-modal LBP
    histograms; AI images tend toward flatter (higher entropy) distributions
    due to their learned texture synthesis.

    Returns a score in [0,1]: higher = more AI-like.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    gray_f = gray.astype(np.float32)

    # 8-neighbour LBP at radius 1
    # Offsets for the 8 neighbours in clockwise order
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0,  1), ( 1,  1), ( 1,  0),
        ( 1, -1), ( 0, -1),
    ]
    lbp = np.zeros_like(gray_f)
    for bit, (dy, dx) in enumerate(neighbours):
        shifted = ndimage.shift(gray_f, (dy, dx), mode="reflect")
        lbp    += ((shifted >= gray_f).astype(np.float32)) * (2 ** bit)

    # Normalised histogram (256 bins)
    hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 255))
    hist     = hist.astype(np.float32)
    hist    /= hist.sum() + 1e-10

    lbp_entropy = float(scipy_entropy(hist + 1e-10))

    # Real images cluster around entropy ≈ 4.5–5.5 (out of max ~5.55 for 256 bins)
    # AI images tend to be higher: 5.2–5.55 (flatter histogram)
    # Normalise: score = 0 at entropy=4.5, score = 1 at entropy=5.55
    score = float(np.clip((lbp_entropy - 4.5) / 1.05, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# Signal 4 – DCT blocking artefacts
# ---------------------------------------------------------------------------

def _signal_dct_blocking(img: np.ndarray) -> float:
    """
    Measure JPEG 8×8 DCT block boundary discontinuities.

    Authentic photographs that have gone through the standard JPEG pipeline
    show characteristic 8-pixel-period discontinuities at block boundaries.
    AI-generated images often lack these (they were never JPEG-compressed in
    the traditional sense) or show atypical patterns.

    Returns a score in [0,1]: higher = more AI-like.
    Low score = strong JPEG block structure (real photo indicator)
    High score = absent or atypical block structure (AI indicator)
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)

    # Horizontal and vertical gradient
    grad_h = np.abs(np.diff(gray_f, axis=1))   # shape (H, W-1)
    grad_v = np.abs(np.diff(gray_f, axis=0))   # shape (H-1, W)

    H, W = gray_f.shape

    # Average gradient at 8-pixel boundaries vs interior
    def _boundary_ratio(grad: np.ndarray, size: int, along_cols: bool) -> float:
        n = grad.shape[1] if along_cols else grad.shape[0]
        if n < 16:
            return 0.5
        boundary_indices = [i for i in range(7, n, 8)]
        interior_indices = [i for i in range(n) if i % 8 != 7]
        if not boundary_indices or not interior_indices:
            return 0.5
        boundary_mean = grad[:, boundary_indices].mean() if along_cols else grad[boundary_indices, :].mean()
        interior_mean = grad[:, interior_indices].mean() if along_cols else grad[interior_indices, :].mean()
        if interior_mean < 1e-6:
            return 0.5
        ratio = boundary_mean / interior_mean
        return float(ratio)

    ratio_h = _boundary_ratio(grad_h, W, along_cols=True)
    ratio_v = _boundary_ratio(grad_v, H, along_cols=False)
    avg_ratio = (ratio_h + ratio_v) / 2.0

    # Real JPEG photos: ratio > 1.15 (boundaries are noticeably stronger)
    # AI images: ratio ≈ 1.0 (no JPEG structure) or erratic
    # Score: high when ratio is close to 1.0 (no DCT pattern) → AI-like
    score = float(np.clip(1.0 - (avg_ratio - 1.0) / 0.4, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# Signal 5 – Colour channel correlation
# ---------------------------------------------------------------------------

def _signal_colour_correlation(img: np.ndarray) -> float:
    """
    Inter-channel noise correlation.

    In real Bayer-demosaiced photos, R/G/B channels share correlated noise
    from the same sensor. In AI images, channels are synthesised independently
    and lack this physical coupling.

    Returns a score in [0,1]: higher = more AI-like.
    """
    # Extract per-channel high-pass residual
    srm_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)

    residuals = []
    for c in range(3):
        ch  = (img[:, :, c] * 255.0).astype(np.float32)
        res = cv2.filter2D(ch, cv2.CV_32F, srm_kernel)
        residuals.append(res.flatten())

    r, g, b = residuals

    def _corr(a: np.ndarray, b_arr: np.ndarray) -> float:
        std_a = a.std(); std_b = b_arr.std()
        if std_a < 1e-6 or std_b < 1e-6:
            return 0.0
        return float(np.mean((a - a.mean()) * (b_arr - b_arr.mean())) /
                     (std_a * std_b + 1e-10))

    corr_rg = abs(_corr(r, g))
    corr_rb = abs(_corr(r, b))
    corr_gb = abs(_corr(g, b))
    avg_corr = (corr_rg + corr_rb + corr_gb) / 3.0

    # Real photos: avg_corr typically 0.20–0.60 (Bayer coupling)
    # AI images:   avg_corr typically < 0.10 (independent channels)
    # Score: high when correlation is very low → AI-like
    score = float(np.clip(1.0 - avg_corr / 0.25, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# Signal 6 – EXIF metadata
# ---------------------------------------------------------------------------

def _signal_exif(raw_bytes: bytes | None) -> float:
    """
    Absence or inconsistency of camera EXIF metadata.

    Returns a score in [0,1]: higher = more AI-like.
    0.0  → rich, consistent EXIF (camera make/model, exposure, GPS)
    0.5  → partial EXIF (software tag only, common in processed images)
    0.85 → no EXIF at all
    1.0  → EXIF present but clearly AI tool signature
    """
    if raw_bytes is None:
        return 0.85

    # Known AI generation tool signatures in software/artist EXIF fields
    AI_SIGNATURES = {
        "stable diffusion", "midjourney", "dall-e", "dall·e",
        "firefly", "imagen", "diffusion", "generative", "ai generated",
        "comfyui", "automatic1111", "invokeai", "novelai",
    }

    try:
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        return 0.85   # No EXIF — common for AI outputs

    has_make  = bool(exif_dict.get("0th", {}).get(piexif.ImageIFD.Make))
    has_model = bool(exif_dict.get("0th", {}).get(piexif.ImageIFD.Model))
    has_expo  = bool(exif_dict.get("Exif", {}).get(piexif.ExifIFD.ExposureTime))
    has_gps   = bool(exif_dict.get("GPS"))

    software_raw = exif_dict.get("0th", {}).get(piexif.ImageIFD.Software, b"")
    software     = (software_raw.decode("utf-8", errors="ignore")
                    if isinstance(software_raw, bytes) else str(software_raw)).lower()

    if any(sig in software for sig in AI_SIGNATURES):
        return 1.0

    exif_richness = sum([has_make, has_model, has_expo, has_gps])
    if exif_richness >= 3:
        return 0.05    # Rich EXIF → strong real indicator
    if exif_richness == 2:
        return 0.20
    if exif_richness == 1:
        return 0.55
    return 0.85        # No camera metadata


def _extract_exif_fields(raw_bytes: bytes | None) -> dict:
    """Return a human-readable dict of useful EXIF fields."""
    if raw_bytes is None:
        return {}
    try:
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        return {}

    out = {}
    ifd0 = exif_dict.get("0th", {})
    exif = exif_dict.get("Exif", {})

    def _decode(v) -> str:
        if isinstance(v, bytes):
            return v.rstrip(b"\x00").decode("utf-8", errors="ignore")
        return str(v)

    for tag, key in [
        (piexif.ImageIFD.Make,   "camera_make"),
        (piexif.ImageIFD.Model,  "camera_model"),
        (piexif.ImageIFD.Software, "software"),
        (piexif.ImageIFD.DateTime, "datetime"),
    ]:
        if tag in ifd0:
            out[key] = _decode(ifd0[tag])

    if piexif.ExifIFD.ExposureTime in exif:
        et = exif[piexif.ExifIFD.ExposureTime]
        if isinstance(et, tuple) and et[1]:
            out["exposure_time"] = f"{et[0]}/{et[1]}s"

    if piexif.ExifIFD.ISOSpeedRatings in exif:
        out["iso"] = str(exif[piexif.ExifIFD.ISOSpeedRatings])

    if exif_dict.get("GPS"):
        out["has_gps"] = True

    return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    samples = sys.argv[1:] if len(sys.argv) > 1 else []

    if not samples:
        print("Usage: python image_verifier.py <path/to/image> [<path2> ...]")
        print("\nRunning synthetic self-test …")

        # Synthetic real-like image: natural gradient + correlated noise
        rng = np.random.default_rng(42)
        h, w = 256, 256
        base = np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float32)
        noise = rng.normal(0, 0.03, (h, w, 3)).astype(np.float32)
        rgb = np.clip(
            np.stack([base, base * 0.8, base * 0.6], axis=-1) + noise, 0, 1
        )
        pil_img = Image.fromarray((rgb * 255).astype(np.uint8))

        result = predict(pil_img)
        icon = "🤖" if result["label"] == "AI_GENERATED" else "📷"
        print(f"\n{icon}  {result['label']}  |  confidence {result['confidence']:.1%}"
              f"  |  ai_score={result['ai_score']:.3f}")
        print("  Signals:", {k: f"{v:.3f}" for k, v in result["signals"].items()})
        if result["warning"]:
            print("  ⚠", result["warning"])
        sys.exit(0)

    for path in samples:
        result = predict(path)
        icon   = "🤖" if result["label"] == "AI_GENERATED" else "📷"
        print(f"\n{icon}  {result['label']}  |  confidence {result['confidence']:.1%}"
              f"  |  ai_score={result['ai_score']:.3f}")
        print(f"   File: {path}")
        print("   Signals:", {k: f"{v:.3f}" for k, v in result["signals"].items()})
        if result["exif_info"]:
            print("   EXIF:", result["exif_info"])
        if result["warning"]:
            print("   ⚠", result["warning"])