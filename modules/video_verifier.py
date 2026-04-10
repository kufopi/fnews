"""
modules/video_verifier.py
Module 3: Deepfake Video Detection (Manipulated Face Detection)

Architecture — multi-signal temporal ensemble.  No external model download.

  Signal 1 · Face Swap Boundary Artefacts (Texture Seam Score)
    Deepfake face swaps paste a generated face onto a background head.
    The boundary region between the swapped face and the neck/hairline
    has characteristic blending artefacts: unnatural texture gradients,
    frequency discontinuities, and statistical anomalies in a narrow
    border band around the detected face oval.

  Signal 2 · Temporal Flickering (Frame-to-Frame Consistency)
    Real faces maintain very stable colour and brightness across
    consecutive frames. GAN and autoencoder deepfakes produce frame-to-frame
    flicker as the generator re-synthesises each frame independently with
    slightly different latent noise. We measure per-face-patch variance
    over time.

  Signal 3 · Facial Landmark Geometry (Symmetry + Physics)
    Mediapipe FaceLandmarker provides 478 landmarks. We check:
      a) Bilateral symmetry score — real faces have slight, stable asymmetry.
           Deepfakes often over-regularise (too symmetric) or show large
           left/right swings across frames.
      b) Eye aspect ratio (EAR) plausibility — deepfakes frequently fail
           to produce natural blinking patterns.
      c) Landmark jitter (normalised velocity) — stable in real faces;
           erratic in deepfakes due to independent per-frame synthesis.

  Signal 4 · Optical Flow Continuity
    Real faces in video produce smooth, physics-consistent optical flow.
    Deepfake stitching produces discontinuous or anomalously uniform flow
    in the face region compared with the background.

  Signal 5 · Colour Space Statistics (Face Region)
    The HSV saturation and hue distribution of a real face is determined
    by skin tone + scene lighting and remains consistent over time.
    Deepfake generators alter these statistics — saturation is often
    artificially boosted and hue variance is elevated.

  Signal 6 · High-Frequency Noise in Face Region
    We run the same SRM filter from image_verifier on face crops.
    Each frame is analysed independently; we report mean and variance
    across frames (high variance = generator instability).

  Ensemble: weighted combination.  Frames are sampled (default 32 frames)
  for speed; the result is aggregated via robust mean (trimmed 10%).

Mediapipe FaceLandmarker usage:
    The new mediapipe ≥0.10 Tasks API requires a .task model bundle.
    This module auto-downloads the lightweight float16 model from the
    official MediaPipe CDN on first use and caches it locally in
    ~/.cache/video_verifier/.
    Fallback: if the download fails (network restricted), we fall back
    to OpenCV Haar cascade face detection for the geometry signals.

Dependencies: cv2, numpy, mediapipe (≥0.10), scipy, PIL
"""

from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from scipy.stats import trim_mean

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR  = Path.home() / ".cache" / "video_verifier"
_MODEL_FILE = _CACHE_DIR / "face_landmarker.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

_DEFAULT_SAMPLE_FRAMES  = 32     # frames to analyse per video
_MIN_FACE_SIZE          = 48     # pixels — smaller faces are skipped
_BORDER_BAND_PX         = 12     # pixels around face oval for seam analysis

_SIGNAL_WEIGHTS = {
    "texture_seam":    0.25,
    "temporal_flicker":0.25,
    "landmark_geometry":0.20,
    "optical_flow":    0.15,
    "colour_stats":    0.10,
    "hf_noise":        0.05,
}

# ---------------------------------------------------------------------------
# Model / face detector management
# ---------------------------------------------------------------------------

_landmarker = None        # cached mediapipe FaceLandmarker
_haar_cascade = None      # fallback OpenCV cascade

def _get_haar_cascade():
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _haar_cascade


def _try_load_landmarker():
    """
    Attempt to load mediapipe FaceLandmarker.  Tries to download the model
    bundle on first use.  Returns the landmarker or None on failure.
    """
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    import mediapipe as mp
    BaseOptions    = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode    = mp.tasks.vision.RunningMode

    # Download model if absent
    if not _MODEL_FILE.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[video_verifier] Downloading FaceLandmarker model → {_MODEL_FILE} …")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILE)
            print("[video_verifier] Model downloaded.")
        except Exception as exc:
            print(f"[video_verifier] Download failed ({exc}). "
                  "Falling back to Haar cascade for geometry signals.")
            return None

    try:
        options = FaceLandmarkerOptions(
            base_options    = BaseOptions(model_asset_path=str(_MODEL_FILE)),
            running_mode    = RunningMode.IMAGE,
            num_faces       = 1,
            min_face_detection_confidence = 0.5,
            min_face_presence_score       = 0.5,
            min_tracking_confidence       = 0.5,
            output_face_blendshapes       = False,
            output_facial_transformation_matrixes = False,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
        print("[video_verifier] FaceLandmarker loaded.")
        return _landmarker
    except Exception as exc:
        print(f"[video_verifier] Could not initialise FaceLandmarker: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(video_input: Union[str, Path]) -> dict:
    """
    Analyse a video file for deepfake face manipulation.

    Parameters
    ----------
    video_input : str | Path
        Path to a video file (MP4, AVI, MOV, MKV …).

    Returns
    -------
    dict with keys:
        label            – "DEEPFAKE" | "REAL"
        confidence       – float in [0,1] for the predicted label
        deepfake_score   – raw ensemble score (higher = more likely deepfake)
        signals          – per-signal sub-scores
        frame_count      – total frames in video
        analysed_frames  – number of frames actually analysed
        faces_detected   – frames in which at least one face was found
        fps              – video frame rate
        duration_sec     – video duration in seconds
        warning          – str | None
    """
    path = Path(video_input)
    if not path.exists():
        return _error_result(f"File not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return _error_result(f"Cannot open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / fps if fps else 0.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < 2:
        cap.release()
        return _error_result("Video has fewer than 2 frames.")

    # Sample frame indices evenly across the video
    n_sample = min(_DEFAULT_SAMPLE_FRAMES, total_frames)
    indices  = np.linspace(0, total_frames - 1, n_sample, dtype=int)

    frames, grey_frames = _read_frames(cap, indices)
    cap.release()

    if not frames:
        return _error_result("Could not decode any frames.")

    landmarker = _try_load_landmarker()

    # Per-frame analysis
    per_frame: list[dict] = []
    for i, (rgb, gray) in enumerate(zip(frames, grey_frames)):
        face_bbox = _detect_face(rgb, gray, landmarker)
        if face_bbox is None:
            continue
        landmarks = _get_landmarks(rgb, landmarker)
        per_frame.append({
            "rgb":       rgb,
            "gray":      gray,
            "bbox":      face_bbox,
            "landmarks": landmarks,
            "frame_idx": i,
        })

    faces_detected = len(per_frame)
    if faces_detected < 2:
        return _error_result(
            f"Fewer than 2 frames contained a detectable face "
            f"({faces_detected}/{len(frames)}). "
            "The video may not contain a close-up face, or face detection failed.",
            partial={
                "frame_count": total_frames,
                "analysed_frames": len(frames),
                "faces_detected": faces_detected,
                "fps": round(fps, 2),
                "duration_sec": round(duration_sec, 2),
            }
        )

    # Compute signals
    signals = {}
    signals["texture_seam"]      = _signal_texture_seam(per_frame)
    signals["temporal_flicker"]  = _signal_temporal_flicker(per_frame)
    signals["landmark_geometry"] = _signal_landmark_geometry(per_frame)
    signals["optical_flow"]      = _signal_optical_flow(per_frame)
    signals["colour_stats"]      = _signal_colour_stats(per_frame)
    signals["hf_noise"]          = _signal_hf_noise(per_frame)

    deepfake_score = float(np.clip(
        sum(signals[k] * _SIGNAL_WEIGHTS[k] for k in signals), 0.0, 1.0
    ))

    label      = "DEEPFAKE" if deepfake_score >= 0.50 else "REAL"
    confidence = deepfake_score if label == "DEEPFAKE" else 1.0 - deepfake_score

    return {
        "label":           label,
        "confidence":      round(confidence, 4),
        "deepfake_score":  round(deepfake_score, 4),
        "signals":         {k: round(v, 4) for k, v in signals.items()},
        "frame_count":     total_frames,
        "analysed_frames": len(frames),
        "faces_detected":  faces_detected,
        "fps":             round(fps, 2),
        "duration_sec":    round(duration_sec, 2),
        "warning":         None,
    }


def confidence_tier(confidence: float) -> str:
    if confidence >= 0.90: return "Very High"
    if confidence >= 0.75: return "High"
    if confidence >= 0.55: return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------

def _read_frames(
    cap: cv2.VideoCapture, indices: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns (rgb_list, gray_list) for the requested frame indices."""
    rgb_frames  = []
    gray_frames = []
    prev_idx    = -1

    for idx in indices:
        if idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb_frames.append(rgb)
        gray_frames.append(gray)
        prev_idx = idx

    return rgb_frames, gray_frames


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

def _detect_face(
    rgb: np.ndarray,
    gray: np.ndarray,
    landmarker,
) -> tuple[int, int, int, int] | None:
    """
    Returns (x, y, w, h) bounding box of the largest detected face, or None.
    Uses Mediapipe if available, else Haar cascade.
    """
    # Try mediapipe first (more accurate)
    if landmarker is not None:
        try:
            import mediapipe as mp
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb.astype(np.uint8)
            )
            result = landmarker.detect(mp_image)
            if result.face_landmarks:
                lm     = result.face_landmarks[0]
                H, W   = rgb.shape[:2]
                xs     = [p.x * W for p in lm]
                ys     = [p.y * H for p in lm]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                w, h   = x2 - x1, y2 - y1
                if w >= _MIN_FACE_SIZE and h >= _MIN_FACE_SIZE:
                    return (x1, y1, w, h)
        except Exception:
            pass

    # Haar cascade fallback
    cascade = _get_haar_cascade()
    if cascade.empty():
        return None
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(_MIN_FACE_SIZE, _MIN_FACE_SIZE)
    )
    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return None
    # Largest face
    largest = max(faces, key=lambda f: f[2] * f[3])
    return tuple(largest)


def _get_landmarks(rgb: np.ndarray, landmarker) -> np.ndarray | None:
    """Returns (N, 2) array of (x,y) landmarks, or None."""
    if landmarker is None:
        return None
    try:
        import mediapipe as mp
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb.astype(np.uint8)
        )
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            H, W = rgb.shape[:2]
            lm   = result.face_landmarks[0]
            pts  = np.array([[p.x * W, p.y * H] for p in lm], dtype=np.float32)
            return pts
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Face crop helper
# ---------------------------------------------------------------------------

def _crop_face(
    img: np.ndarray, bbox: tuple, pad: float = 0.0
) -> np.ndarray | None:
    x, y, w, h = bbox
    H, W = img.shape[:2]
    px = int(w * pad); py = int(h * pad)
    x1 = max(0, x - px); y1 = max(0, y - py)
    x2 = min(W, x + w + px); y2 = min(H, y + h + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Signal 1 – Texture seam (face boundary artefacts)
# ---------------------------------------------------------------------------

def _signal_texture_seam(frames: list[dict]) -> float:
    """
    Measure texture discontinuity at the face oval boundary.
    High score = sharp unnatural edge → deepfake indicator.
    """
    scores = []
    for f in frames:
        rgb, bbox = f["rgb"], f["bbox"]
        x, y, w, h = bbox
        H_img, W_img = rgb.shape[:2]

        if w < _MIN_FACE_SIZE or h < _MIN_FACE_SIZE:
            continue

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Create elliptical mask for face oval
        mask_inner = np.zeros(gray.shape, dtype=np.uint8)
        cx = x + w // 2; cy = y + h // 2
        cv2.ellipse(mask_inner, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, -1)

        # Outer ring (border band)
        mask_outer = np.zeros(gray.shape, dtype=np.uint8)
        cv2.ellipse(mask_outer, (cx, cy),
                    (w // 2 + _BORDER_BAND_PX, h // 2 + _BORDER_BAND_PX),
                    0, 0, 360, 255, -1)
        border_mask = cv2.subtract(mask_outer, mask_inner)

        if border_mask.sum() == 0:
            continue

        # Laplacian as texture indicator
        lap = cv2.Laplacian(gray, cv2.CV_32F)

        inside_lap  = lap[mask_inner  > 0].std()
        border_lap  = lap[border_mask > 0].std()
        outside_lap = lap[(mask_outer == 0)].std()
        # Normalised in range [0,1] — real: border ≈ inside; fake: border >> inside
        if inside_lap < 1e-6 or outside_lap < 1e-6:
            scores.append(0.5)
            continue

        ratio = border_lap / (0.5 * (inside_lap + outside_lap) + 1e-6)
        # ratio ~1.0 is normal; ratio > 1.5 is suspicious
        score = float(np.clip((ratio - 1.0) / 1.0, 0.0, 1.0))
        scores.append(score)

    if not scores:
        return 0.5
    return float(trim_mean(scores, 0.1))


# ---------------------------------------------------------------------------
# Signal 2 – Temporal flickering
# ---------------------------------------------------------------------------

def _signal_temporal_flicker(frames: list[dict]) -> float:
    """
    Frame-to-frame brightness / colour variance in the face region.
    Deepfake generators produce per-frame independent noise → flicker.
    """
    face_means = []
    for f in frames:
        crop = _crop_face(f["rgb"], f["bbox"], pad=0.05)
        if crop is None or crop.size == 0:
            continue
        face_means.append(crop.astype(np.float32).mean(axis=(0, 1)))  # (3,)

    if len(face_means) < 3:
        return 0.5

    arr   = np.array(face_means)          # (N, 3)
    diffs = np.diff(arr, axis=0)          # (N-1, 3) frame-to-frame deltas
    # Mean absolute change per channel
    mac   = np.abs(diffs).mean(axis=0)    # (3,)
    mean_mac = float(mac.mean())

    # Real videos: mean_mac < 2.0 (255-scale)
    # Deepfakes:   mean_mac can be 4–15 due to generator flicker
    score = float(np.clip((mean_mac - 1.5) / 8.0, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# Signal 3 – Landmark geometry
# ---------------------------------------------------------------------------

def _signal_landmark_geometry(frames: list[dict]) -> float:
    """
    Facial landmark plausibility and temporal stability.
    Analyses symmetry, eye aspect ratio, and landmark jitter.
    """
    frames_with_lm = [f for f in frames if f.get("landmarks") is not None]
    if not frames_with_lm:
        return 0.5   # Landmarker unavailable — neutral

    symmetry_scores = []
    ear_scores      = []
    jitter_scores   = []

    prev_lm = None
    for f in frames_with_lm:
        lm = f["landmarks"]   # (478, 2)
        H, W = f["rgb"].shape[:2]

        # ── Symmetry (compare left vs right eye outer landmarks) ──────────
        # Mediapipe landmark indices: left eye outer=33, right eye outer=263
        # (mirrored in face space)
        if len(lm) > 263:
            left_eye  = lm[33]
            right_eye = lm[263]
            nose_tip  = lm[4]
            dist_l = np.linalg.norm(left_eye  - nose_tip)
            dist_r = np.linalg.norm(right_eye - nose_tip)
            sym_ratio = min(dist_l, dist_r) / (max(dist_l, dist_r) + 1e-6)
            # Real: sym_ratio 0.85–0.98 (slight natural asymmetry)
            # Deepfake over-regularised: sym_ratio > 0.98 OR wildly asymmetric < 0.7
            if sym_ratio > 0.98:
                symmetry_scores.append(0.7)   # Too symmetric → AI regularisation
            elif sym_ratio < 0.70:
                symmetry_scores.append(0.8)   # Too asymmetric → poor stitching
            else:
                symmetry_scores.append(0.1)   # Natural range

        # ── Eye Aspect Ratio (EAR) ────────────────────────────────────────
        # Left eye landmarks: 33(outer), 160, 158(top), 133, 153, 144(bot)
        # (simplified using 4 points: top/bottom of upper/lower lid)
        if len(lm) > 160:
            # Approximate EAR using available landmark indices
            p1, p2 = lm[159], lm[145]   # upper/lower lid (left eye)
            p3, p4 = lm[33],  lm[133]   # inner/outer corner (left eye)
            eye_h  = np.linalg.norm(p1 - p2)
            eye_w  = np.linalg.norm(p3 - p4)
            ear    = eye_h / (eye_w + 1e-6)
            # Natural EAR range 0.15–0.40; deepfakes often produce 0.0 (frozen)
            # or > 0.50 (over-wide eyes)
            if ear < 0.05 or ear > 0.50:
                ear_scores.append(0.75)
            else:
                ear_scores.append(0.05)

        # ── Landmark jitter (velocity) ────────────────────────────────────
        if prev_lm is not None:
            velocity  = np.linalg.norm(lm - prev_lm, axis=1)  # per-landmark
            mean_vel  = velocity.mean()
            face_diag = float(np.linalg.norm([W, H]))
            norm_vel  = mean_vel / (face_diag + 1e-6)
            # Real: norm_vel 0.001–0.010 (small natural motion)
            # Deepfake: norm_vel can be > 0.020 (per-frame regeneration)
            jitter_scores.append(float(np.clip((norm_vel - 0.008) / 0.02, 0.0, 1.0)))
        prev_lm = lm

    sub = []
    if symmetry_scores: sub.append(np.mean(symmetry_scores))
    if ear_scores:       sub.append(np.mean(ear_scores))
    if jitter_scores:    sub.append(np.mean(jitter_scores))

    return float(np.mean(sub)) if sub else 0.5


# ---------------------------------------------------------------------------
# Signal 4 – Optical flow continuity
# ---------------------------------------------------------------------------

def _signal_optical_flow(frames: list[dict]) -> float:
    """
    Measure optical flow anomaly: face region vs background.
    In deepfakes, the face region often moves inconsistently with the background.
    """
    if len(frames) < 3:
        return 0.5

    flow_ratios = []
    for i in range(1, len(frames)):
        prev_g = frames[i-1]["gray"]
        curr_g = frames[i]["gray"]
        bbox   = frames[i]["bbox"]

        H, W = prev_g.shape

        flow = cv2.calcOpticalFlowFarneback(
            prev_g.astype(np.uint8), curr_g.astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        x, y, w, h = bbox
        x2, y2 = min(W, x+w), min(H, y+h)
        if x2 <= x or y2 <= y:
            continue

        face_flow = flow_mag[y:y2, x:x2]
        # Background mask: everything outside expanded face region
        bg_mask        = np.ones((H, W), dtype=bool)
        pad            = 20
        bg_mask[max(0,y-pad):min(H,y2+pad), max(0,x-pad):min(W,x2+pad)] = False
        bg_flow        = flow_mag[bg_mask]

        if face_flow.size == 0 or bg_flow.size == 0:
            continue

        face_mean = face_flow.mean()
        bg_mean   = bg_flow.mean()

        # In deepfakes the face is often more static than the background
        # (generator re-synthesis vs natural head motion), or vice versa
        if bg_mean < 0.1:
            flow_ratios.append(0.2)   # static scene — not informative
            continue

        ratio = face_mean / (bg_mean + 1e-6)
        # Real: ratio 0.7–1.5 (face and body move similarly)
        # Deepfake: ratio < 0.4 (face frozen) or > 2.0 (face moving unnaturally)
        anomaly = float(np.clip(abs(np.log(ratio + 1e-6)) / 1.0, 0.0, 1.0))
        flow_ratios.append(anomaly)

    if not flow_ratios:
        return 0.5
    return float(trim_mean(flow_ratios, 0.1))


# ---------------------------------------------------------------------------
# Signal 5 – Colour statistics in face region
# ---------------------------------------------------------------------------

def _signal_colour_stats(frames: list[dict]) -> float:
    """
    HSV saturation consistency over time in the face region.
    Deepfakes often over-saturate or have high hue variance.
    """
    sat_vals  = []
    hue_stds  = []

    for f in frames:
        crop = _crop_face(f["rgb"], f["bbox"], pad=0.05)
        if crop is None or crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_vals.append(hsv[:, :, 1].mean())   # mean saturation
        hue_stds.append(hsv[:, :, 0].std())    # hue std dev

    if not sat_vals:
        return 0.5

    mean_sat    = float(np.mean(sat_vals))
    sat_var     = float(np.std(sat_vals))     # inter-frame saturation variance
    mean_hue_sd = float(np.mean(hue_stds))

    # Real: mean_sat 60–160; sat_var < 8; mean_hue_sd 8–25
    # Deepfake indicators: sat_var > 15 (flicker) or mean_sat > 180 (over-saturated)
    var_score = float(np.clip((sat_var - 5.0) / 15.0, 0.0, 1.0))
    sat_score = float(np.clip((mean_sat - 150.0) / 80.0, 0.0, 1.0))
    hue_score = float(np.clip((mean_hue_sd - 30.0) / 20.0, 0.0, 1.0))

    return (var_score * 0.5 + sat_score * 0.3 + hue_score * 0.2)


# ---------------------------------------------------------------------------
# Signal 6 – High-frequency noise in face region
# ---------------------------------------------------------------------------

def _signal_hf_noise(frames: list[dict]) -> float:
    """
    SRM noise residual variance across frames in face region.
    High inter-frame noise variance = generator instability → deepfake.
    """
    srm = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    noise_stds = []

    for f in frames:
        crop = _crop_face(f["rgb"], f["bbox"], pad=0.0)
        if crop is None or crop.size == 0:
            continue
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        residual  = cv2.filter2D(gray_crop, cv2.CV_32F, srm)
        noise_stds.append(residual.std())

    if len(noise_stds) < 2:
        return 0.5

    mean_std = float(np.mean(noise_stds))
    var_std  = float(np.std(noise_stds))    # inter-frame variance of noise

    # Real: var_std < 2.0 (stable noise floor from sensor)
    # Deepfake: var_std > 4.0 (flickering synthetic noise)
    score = float(np.clip((var_std - 1.5) / 5.0, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _error_result(msg: str, partial: dict | None = None) -> dict:
    base = {
        "label": "UNKNOWN", "confidence": 0.0, "deepfake_score": 0.5,
        "signals": {}, "frame_count": 0, "analysed_frames": 0,
        "faces_detected": 0, "fps": 0.0, "duration_sec": 0.0,
        "warning": msg,
    }
    if partial:
        base.update(partial)
    return base


# ---------------------------------------------------------------------------
# CLI smoke test  (creates a synthetic video in /tmp and analyses it)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        print("Usage: python video_verifier.py <path/to/video> [<path2> ...]")
        print("\nRunning synthetic self-test (generating temp video) …")

        # Create a synthetic 3-second 25fps video with a moving coloured disc
        # (face detection will likely fail — that's expected; test the pipeline)
        fps_out = 25
        n_frames = fps_out * 3
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        out = cv2.VideoWriter(tmp_path, fourcc, fps_out, (320, 240))
        for i in range(n_frames):
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 60
            cx = int(160 + 60 * np.sin(2 * np.pi * i / fps_out))
            cy = 120
            cv2.circle(frame, (cx, cy), 40, (180, 120, 90), -1)   # skin-like disc
            out.write(frame)
        out.release()

        result = predict(tmp_path)
        os.unlink(tmp_path)

        if result["label"] == "UNKNOWN":
            print(f"\n⚠  Pipeline note: {result['warning']}")
            print(f"   (Expected — synthetic disc ≠ real face; face detector skipped.)")
        else:
            icon = "🎭" if result["label"] == "DEEPFAKE" else "✅"
            print(f"\n{icon}  {result['label']}  |  confidence {result['confidence']:.1%}"
                  f"  |  deepfake_score={result['deepfake_score']:.3f}")
        print(f"\n   Frames total/analysed/faces: "
              f"{result['frame_count']} / {result['analysed_frames']} / {result['faces_detected']}")
        print("   FPS:", result["fps"], "  Duration:", result["duration_sec"], "s")
        sys.exit(0)

    for path in paths:
        result = predict(path)
        if result["label"] == "UNKNOWN":
            print(f"\n⚠  {path}: {result['warning']}")
            continue
        icon = "🎭" if result["label"] == "DEEPFAKE" else "✅"
        print(f"\n{icon}  {result['label']}  |  confidence {result['confidence']:.1%}"
              f"  |  deepfake_score={result['deepfake_score']:.3f}")
        print(f"   File: {path}")
        print("   Signals:", {k: f"{v:.3f}" for k, v in result["signals"].items()})
        if result["warning"]:
            print("   ⚠", result["warning"])