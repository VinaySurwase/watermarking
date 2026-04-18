"""
analysis_view.py
────────────────
Robustness analysis — applies various attacks to a watermarked image
and reports whether each attacked version is detected as tampered.

Endpoints
─────────
POST  watermarking/analysis/baseline/
  → Run baseline (no-attack) tamper check.

POST  watermarking/analysis/attack/
  → Run a single named attack, return metrics + attacked image (base64).

POST  watermarking/analysis/       (legacy batch — still works)
"""

import os, tempfile, uuid, traceback, base64

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404

from .models import ImageProcess
from .utility import load_key, getImg, reconstruct_full_image
from .embedding import _forward_pipeline
from .path_helpers import path_key


# ═════════════════════════════════════════════════════════════════════
#  ATTACK FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def _gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float64)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _salt_pepper_noise(img, amount=0.02):
    out = img.copy()
    total = img.shape[0] * img.shape[1]
    n_salt = int(total * amount / 2)
    n_pepper = int(total * amount / 2)
    for _ in range(n_salt):
        y, x = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
        out[y, x] = 255
    for _ in range(n_pepper):
        y, x = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
        out[y, x] = 0
    return out


def _median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)


def _gaussian_lowpass(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _jpeg_compress(img, quality=50):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _jpeg2000_compress(img, ratio=48):
    _, buf = cv2.imencode('.jp2', img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, ratio])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _rotation_attack(img, angle=5):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _scaling_attack(img, scale=0.8):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_LANCZOS4)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _translation_attack(img, tx=10, ty=10):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _sharpening_attack(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def _combined_attack(img):
    out = _gaussian_noise(img, sigma=15)
    out = _jpeg_compress(out, quality=70)
    out = _median_filter(out, ksize=3)
    return out


# Registry: key → (display_name, function, description)
ATTACKS = {
    "gaussian_noise":       ("Gaussian Noise",       _gaussian_noise,
                             "Additive Gaussian noise (σ=25)"),
    "salt_pepper":          ("Salt & Pepper Noise",  _salt_pepper_noise,
                             "Random salt-and-pepper noise (2% pixels)"),
    "median_filter":        ("Median Filtering",     _median_filter,
                             "3×3 median filter"),
    "gaussian_lowpass":     ("Gaussian Low-pass",    _gaussian_lowpass,
                             "5×5 Gaussian blur"),
    "jpeg_compression":     ("JPEG Compression",     _jpeg_compress,
                             "JPEG quality = 50"),
    "jpeg2000_compression": ("JPEG2000 Compression", _jpeg2000_compress,
                             "JPEG2000 compression"),
    "rotation":             ("Rotation Attack",      _rotation_attack,
                             "5° rotation with reflection padding"),
    "scaling":              ("Scaling Attack",       _scaling_attack,
                             "Scale down to 80% then back up"),
    "translation":          ("Translation Attack",   _translation_attack,
                             "Shift by (10, 10) px with reflection"),
    "sharpening":           ("Sharpening Attack",    _sharpening_attack,
                             "Laplacian sharpening kernel"),
    "combined":             ("Combined Attacks",     _combined_attack,
                             "Gaussian noise + JPEG + median filter"),
}

# Ordered keys for frontend iteration
ATTACK_ORDER = [
    "gaussian_noise", "salt_pepper", "median_filter", "gaussian_lowpass",
    "jpeg_compression", "jpeg2000_compression", "rotation", "scaling",
    "translation", "sharpening", "combined",
]


# ═════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════

def _quick_tamper_check(image_path, key):
    img = getImg(image_path)
    img = reconstruct_full_image(img, key)
    (_, HSw_hat_list, _, _, positions,
     LL, highpasses, tr) = _forward_pipeline(img, key.block_size, key.dtcwt_levels)
    n_blocks = len(HSw_hat_list)
    T = key.tamper_threshold
    sv_deltas = np.array(
        [abs(HSw_hat_list[i][0] - key.HSw_new_dominant[i]) for i in range(n_blocks)],
        dtype=np.float64,
    )
    tamper_flat = sv_deltas > T
    LL_shape = LL.shape
    n_rows = LL_shape[0] // key.block_size
    n_cols = LL_shape[1] // key.block_size
    n_used = n_rows * n_cols
    n_tampered = int(tamper_flat[:n_used].sum())
    tampered_frac = n_tampered / max(n_used, 1)
    return bool(n_tampered > 0), tampered_frac, n_tampered, n_used


def _pixel_diff(original_bgr, attacked_bgr):
    a, b = original_bgr, attacked_bgr
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    diff = np.any(a != b, axis=2)
    changed = int(np.sum(diff))
    total = a.shape[0] * a.shape[1]
    return changed, total, round(changed / total * 100, 4)


def _compute_psnr(img1, img2):
    """Compute PSNR between two BGR images. Returns None when identical."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(img2, (a.shape[1], int(a.shape[0])),
                       interpolation=cv2.INTER_LANCZOS4).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return None  # identical images — inf is not valid JSON
    return round(float(10 * np.log10(255.0 ** 2 / mse)), 4)


def _compute_nc(img1, img2):
    """Normalized Cross-Correlation between two BGR images."""
    a = img1.astype(np.float64).flatten()
    b = img2.astype(np.float64)
    if img1.shape != img2.shape:
        b = cv2.resize(img2, (img1.shape[1], img1.shape[0]),
                       interpolation=cv2.INTER_LANCZOS4).astype(np.float64)
    b = b.flatten()
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if den == 0:
        return 0.0
    return round(float(num / den), 6)


def _img_to_base64(img_bgr, max_dim=512):
    """Encode a BGR image as a JPEG base64 data-URI, downscaling if needed."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"


def _save_temp_upload(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1] or ".png"
    p = os.path.join(tempfile.gettempdir(), f"ana_{uuid.uuid4().hex}{ext}")
    with open(p, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return p


def _save_temp_array(img_bgr, suffix=".png"):
    p = os.path.join(tempfile.gettempdir(), f"ana_{uuid.uuid4().hex}{suffix}")
    cv2.imwrite(p, img_bgr)
    return p


def _validate_and_load(request):
    """Common validation for all analysis endpoints.
    Returns (wm_file, process_id, source, key, error_response)."""
    wm_file = request.FILES.get("watermarked_image")
    process_id = request.data.get("process_id")

    if not wm_file:
        return None, None, None, None, Response(
            {"error": "watermarked_image is required."},
            status=status.HTTP_400_BAD_REQUEST)
    if not process_id:
        return None, None, None, None, Response(
            {"error": "process_id is required."},
            status=status.HTTP_400_BAD_REQUEST)

    allowed = {"image/png", "image/jpeg", "image/jpg"}
    if wm_file.content_type not in allowed:
        return None, None, None, None, Response(
            {"error": "Invalid image format."},
            status=status.HTTP_400_BAD_REQUEST)

    source = get_object_or_404(ImageProcess, id=process_id, user=request.user)
    if source.status != ImageProcess.Status.COMPLETED:
        return None, None, None, None, Response(
            {"error": "Embedding process not completed."},
            status=status.HTTP_400_BAD_REQUEST)

    key_path = path_key(source) + ".npz"
    if not os.path.exists(key_path):
        return None, None, None, None, Response(
            {"error": "Key file not found."},
            status=status.HTTP_404_NOT_FOUND)

    key = load_key(key_path)
    return wm_file, process_id, source, key, None


# ═════════════════════════════════════════════════════════════════════
#  ENDPOINT: attack list
# ═════════════════════════════════════════════════════════════════════

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_attack_list(request):
    """Return the ordered list of available attacks."""
    items = []
    for k in ATTACK_ORDER:
        name, fn, desc = ATTACKS[k]
        items.append({"attack_key": k, "attack_name": name, "description": desc})
    return Response({"attacks": items})


# ═════════════════════════════════════════════════════════════════════
#  ENDPOINT: baseline check
# ═════════════════════════════════════════════════════════════════════

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_baseline(request):
    """Run a no-attack tamper check on the watermarked image."""
    wm_file, process_id, source, key, err = _validate_and_load(request)
    if err:
        return err

    tmp_paths = []
    try:
        wm_path = _save_temp_upload(wm_file)
        tmp_paths.append(wm_path)

        detected, frac, n_tampered, n_blocks = _quick_tamper_check(wm_path, key)

        return Response({
            "tamper_detected": detected,
            "tampered_blocks": n_tampered,
            "total_blocks": n_blocks,
            "tampered_fraction": round(frac * 100, 2),
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ═════════════════════════════════════════════════════════════════════
#  ENDPOINT: single attack
# ═════════════════════════════════════════════════════════════════════

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_single_attack(request):
    """
    POST  watermarking/analysis/attack/

    Form-data:
      • watermarked_image  – file
      • process_id         – int
      • attack_key         – string (one of ATTACK_ORDER)

    Returns metrics + attacked image as base64 thumbnail.
    """
    attack_key = request.data.get("attack_key")
    if not attack_key or attack_key not in ATTACKS:
        return Response(
            {"error": f"Invalid attack_key. Choose from: {ATTACK_ORDER}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    wm_file, process_id, source, key, err = _validate_and_load(request)
    if err:
        return err

    atk_name, atk_fn, atk_desc = ATTACKS[attack_key]
    tmp_paths = []

    try:
        wm_path = _save_temp_upload(wm_file)
        tmp_paths.append(wm_path)

        wm_bgr = cv2.imread(wm_path)
        if wm_bgr is None:
            return Response({"error": "Could not decode image."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Apply attack
        attacked_bgr = atk_fn(wm_bgr.copy())

        # Pixel diff
        px_changed, px_total, px_pct = _pixel_diff(wm_bgr, attacked_bgr)

        # PSNR & NC
        psnr = _compute_psnr(wm_bgr, attacked_bgr)
        nc = _compute_nc(wm_bgr, attacked_bgr)

        # Attacked image as base64 thumbnail
        attacked_b64 = _img_to_base64(attacked_bgr)

        # Save attacked image for tamper check
        atk_path = _save_temp_array(attacked_bgr)
        tmp_paths.append(atk_path)

        # Tamper check
        detected, frac, n_tampered, n_blocks = _quick_tamper_check(atk_path, key)

        # Guard non-finite values for JSON safety
        def _safe(v, decimals=4):
            if v is None:
                return None
            if not np.isfinite(v):
                return None
            return round(float(v), decimals)

        return Response({
            "attack_key": attack_key,
            "attack_name": atk_name,
            "description": atk_desc,
            "pixels_changed": px_changed,
            "percent_changed": px_pct,
            "psnr": _safe(psnr, 4),
            "nc": _safe(nc, 6),
            "attacked_image": attacked_b64,
            "tamper_detected": detected,
            "tampered_blocks": n_tampered,
            "total_blocks": n_blocks,
            "tampered_fraction": round(frac * 100, 2),
            "status": "success",
        })

    except Exception as e:
        traceback.print_exc()
        return Response({
            "attack_key": attack_key,
            "attack_name": atk_name,
            "description": atk_desc,
            "status": "error",
            "error": str(e),
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ═════════════════════════════════════════════════════════════════════
#  LEGACY BATCH ENDPOINT (kept for backward compat)
# ═════════════════════════════════════════════════════════════════════

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_analysis(request):
    """Batch endpoint — runs all attacks at once."""
    wm_file, process_id, source, key, err = _validate_and_load(request)
    if err:
        return err

    tmp_paths = []
    try:
        wm_path = _save_temp_upload(wm_file)
        tmp_paths.append(wm_path)

        wm_bgr = cv2.imread(wm_path)
        if wm_bgr is None:
            return Response({"error": "Could not decode image."},
                            status=status.HTTP_400_BAD_REQUEST)

        baseline_detected, baseline_frac, baseline_tampered, total_blocks = \
            _quick_tamper_check(wm_path, key)

        results = []
        for atk_key in ATTACK_ORDER:
            atk_name, atk_fn, atk_desc = ATTACKS[atk_key]
            try:
                attacked_bgr = atk_fn(wm_bgr.copy())
                px_changed, px_total, px_pct = _pixel_diff(wm_bgr, attacked_bgr)
                atk_path = _save_temp_array(attacked_bgr)
                tmp_paths.append(atk_path)
                detected, frac, n_tampered, n_blocks = _quick_tamper_check(atk_path, key)
                results.append({
                    "attack_key": atk_key, "attack_name": atk_name,
                    "description": atk_desc, "pixels_changed": px_changed,
                    "total_pixels": px_total, "percent_changed": px_pct,
                    "tamper_detected": detected, "tampered_blocks": n_tampered,
                    "total_blocks": n_blocks,
                    "tampered_fraction": round(frac * 100, 2),
                    "status": "success",
                })
            except Exception as atk_err:
                traceback.print_exc()
                results.append({
                    "attack_key": atk_key, "attack_name": atk_name,
                    "description": atk_desc, "status": "error",
                    "error": str(atk_err),
                })

        return Response({
            "baseline": {
                "tamper_detected": baseline_detected,
                "tampered_blocks": baseline_tampered,
                "total_blocks": total_blocks,
                "tampered_fraction": round(baseline_frac * 100, 2),
            },
            "attacks": results,
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass
