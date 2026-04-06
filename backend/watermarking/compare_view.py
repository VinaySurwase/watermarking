"""
Compare View – computes image-quality metrics between an original
host image and its watermarked counterpart.

Metrics computed:
  • PSNR  (Peak Signal-to-Noise Ratio, dB)
  • MSE   (Mean Squared Error)
  • NC    (Normalized Cross-Correlation)
  • SSIM  (Structural Similarity Index)
  • BER   (Bit Error Rate)
"""

import os
import tempfile
import uuid

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


# ─── Metric helpers ───────────────────────────────────────────────

def _to_grayscale(img):
    """Convert BGR image to grayscale (single-channel uint8)."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _compute_mse(a, b):
    """Mean Squared Error between two same-shape arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def _compute_psnr(a, b):
    """Peak Signal-to-Noise Ratio (dB). Returns Inf when images are identical."""
    mse = _compute_mse(a, b)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _compute_ssim(a, b):
    """Structural Similarity Index (scalar)."""
    val, _ = ssim(a, b, full=True)
    return float(val)


def _compute_nc(a, b):
    """Normalized Cross-Correlation (1.0 = identical)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


def _compute_ber(a, b):
    """
    Bit Error Rate – fraction of differing bits when both images are
    thresholded to binary (Otsu).  Useful for watermark logos.
    """
    _, bin_a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bits_a = (bin_a > 0).astype(np.uint8)
    bits_b = (bin_b > 0).astype(np.uint8)

    total_bits = bits_a.size
    error_bits = int(np.sum(bits_a != bits_b))

    return float(error_bits / total_bits) if total_bits > 0 else 0.0


# ─── Helper: save an InMemoryUploadedFile to a temp path ──────────

def _save_temp(uploaded_file):
    """Write an UploadedFile to a temp file and return the path."""
    ext = os.path.splitext(uploaded_file.name)[1] or ".png"
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"cmp_{uuid.uuid4().hex}{ext}",
    )
    with open(tmp_path, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return tmp_path


# ─── API endpoint ─────────────────────────────────────────────────

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def compare_images(request):
    """
    POST  watermarking/compare/
    Form-data fields:
      • original_image   – the original (host) image file
      • watermarked_image – the watermarked image file

    Returns JSON with PSNR, MSE, NC, SSIM, BER.
    """
    original_file = request.FILES.get("original_image")
    watermarked_file = request.FILES.get("watermarked_image")

    if not original_file or not watermarked_file:
        return Response(
            {"error": "Both original_image and watermarked_image are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    allowed_types = {"image/png", "image/jpeg", "image/jpg"}
    if original_file.content_type not in allowed_types:
        return Response(
            {"error": "Invalid original image format. Use PNG or JPEG."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if watermarked_file.content_type not in allowed_types:
        return Response(
            {"error": "Invalid watermarked image format. Use PNG or JPEG."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    tmp_orig = None
    tmp_wm = None

    try:
        # Save to temp files so OpenCV can read them
        tmp_orig = _save_temp(original_file)
        tmp_wm = _save_temp(watermarked_file)

        orig = cv2.imread(tmp_orig)
        wm = cv2.imread(tmp_wm)

        if orig is None:
            return Response(
                {"error": "Could not decode the original image."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if wm is None:
            return Response(
                {"error": "Could not decode the watermarked image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Resize watermarked to match original if dimensions differ
        if orig.shape[:2] != wm.shape[:2]:
            wm = cv2.resize(wm, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # Convert to grayscale for metric computation
        gray_orig = _to_grayscale(orig)
        gray_wm = _to_grayscale(wm)

        # Compute all metrics
        mse_val = _compute_mse(gray_orig, gray_wm)
        psnr_val = _compute_psnr(gray_orig, gray_wm)
        ssim_val = _compute_ssim(gray_orig, gray_wm)
        nc_val = _compute_nc(gray_orig, gray_wm)
        ber_val = _compute_ber(gray_orig, gray_wm)

        return Response({
            "psnr": round(psnr_val, 4),
            "mse": round(mse_val, 4),
            "nc": round(nc_val, 6),
            "ssim": round(ssim_val, 6),
            "ber": round(ber_val, 6),
            "dimensions": {
                "original": list(orig.shape[:2]),
                "watermarked": list(wm.shape[:2]),
            },
        })

    except Exception as e:
        return Response(
            {"error": f"Comparison failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        # Clean up temp files
        for p in (tmp_orig, tmp_wm):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
