"""
extract_watermark.py
────────────────────
Implements Algorithm 2 — watermark extraction from a watermarked image.

Public entry point
──────────────────
    extract_watermark(watermarked_path, key, output_path, extraction_id)

Algorithm
---------
  1. Load & resize the watermarked image  (same resize() as embedding)
  2. Forward pipeline → HSw_hat_list      (per-block dominant SVs)
  3. Recover watermark SVs:
         Sw' = (HSw_hat - HSw) / α*
  4. Average SVs → single reconstruction via ISVD
  5. Henon decrypt  →  W_extracted
  6. Normalise to [0, 255] and save
"""

import os
import traceback

import cv2
import numpy as np
from PIL import Image

from .embedding import (
    EmbedKey,
    _forward_pipeline,
    _isvd,
    henon_decrypt,
    resize,
)
from .models import WatermarkExtraction
from .utility import reconstruct_full_image,getImg

# ═════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _save_png(path: str, img: np.ndarray) -> None:
    """Write a uint8 numpy array as PNG, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"cv2.imwrite failed: {path}")


def _tick(extraction: "WatermarkExtraction | None", progress: int) -> None:
    if extraction is not None:
        extraction.set_status(WatermarkExtraction.Status.EXTRACTING, progress)


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def extract_watermark(
    watermarked_path : str,
    key              : EmbedKey,
    output_path      : str = "extracted_watermark.png",
    extraction_id    : int | None = None,
) -> np.ndarray:
    """
    Recover the embedded watermark from *watermarked_path* using *key*.

    Parameters
    ----------
    watermarked_path : path to the watermarked image (PNG / JPEG)
    key              : EmbedKey produced at embedding time
    output_path      : where to save the extracted watermark PNG
    extraction_id    : optional WatermarkExtraction PK – enables live
                       DB progress updates and metadata write-back

    Returns
    -------
    W_out : uint8 ndarray of the recovered watermark
    """

   
    extraction: WatermarkExtraction | None = None
    if extraction_id is not None:
        try:
            extraction = WatermarkExtraction.objects.get(id=extraction_id)
        except WatermarkExtraction.DoesNotExist:
            extraction = None

    # =====================================================
    # 🔹 STEP 1: load & resize watermarked image
    # =====================================================
    _tick(extraction, 10)

    Iw = getImg(watermarked_path)
    Iw = reconstruct_full_image(Iw,key)
    # =====================================================
    # 🔹 STEP 2: forward pipeline → per-block SVs
    # =====================================================
    _tick(extraction, 25)

    _, HSw_hat_list, _, _, positions, LL, highpasses, tr = _forward_pipeline(Iw, key.block_size, key.dtcwt_levels)

    n_blocks = len(HSw_hat_list)
    sv_len   = len(HSw_hat_list[0])

    # =====================================================
    # 🔹 STEP 3: recover watermark singular values
    # =====================================================
    _tick(extraction, 45)

    HSw_hat = np.array(HSw_hat_list)              
    HSw     = np.array(key.HSw_list)              
    Sw_prime = (HSw_hat - HSw) / key.alpha_star   

    # =====================================================
    # 🔹 STEP 4: average → single ISVD reconstruction
    # =====================================================
    _tick(extraction, 60)

    sv_mean = np.mean(Sw_prime, axis=0)                   

    k = min(key.Uw.shape[1], key.Vtw.shape[0])
    sv_k = np.zeros(k)
    sv_k[:min(k, len(sv_mean))] = sv_mean[:k]

    Cw = _isvd(key.Uw[:, :k], sv_k, key.Vtw[:k, :])   

    # =====================================================
    # 🔹 STEP 5: Henon decrypt
    # =====================================================
    _tick(extraction, 75)

    W_ext = henon_decrypt(Cw, a=key.henon_a, b=key.henon_b)

    # =====================================================
    # 🔹 STEP 6: normalise & save
    # =====================================================
    _tick(extraction, 88)

    W_norm = W_ext - W_ext.min()
    if W_norm.max() > 0:
        W_norm /= W_norm.max()
    W_out = (W_norm * 255).astype(np.uint8)

    _save_png(output_path, W_out)

    # =====================================================
    # 🔹 STEP 7: write metadata back to DB row
    # =====================================================
    if extraction is not None:
        extraction.alpha_star      = float(key.alpha_star)
        extraction.n_blocks        = int(n_blocks)
        extraction.sv_length       = int(sv_len)
        extraction.watermark_shape = list(W_out.shape)
        extraction.set_status(WatermarkExtraction.Status.COMPLETED, 100)

    return W_out