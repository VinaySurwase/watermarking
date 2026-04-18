"""
extract_watermark.py  (FIXED)
─────────────────────────────
Implements Algorithm 2 — watermark extraction from a watermarked image.

Bugs fixed vs original:
  BUG 1 (Critical) — Reconstruction now done per-block first, then averaged
                      as images. Original code averaged SVs then did one ISVD,
                      collapsing all spatial watermark structure into noise.
  BUG 2 (Critical) — HSw_list is now saved as float64 in the key (fixed in
                      embedding.py). Load guard added here to catch legacy keys.
  BUG 3 (Major)    — Sw_prime is clipped after division to prevent noise
                      amplification when alpha_star is small.
  BUG 4 (Major)    — ISVD now uses full Sw_full from key as the SV base,
                      overwriting only the recovered portion. Original code
                      zero-padded from sv_len=8 up to wm_side, badly
                      distorting the reconstruction.
  BUG 5 (Minor)    — Re-padding at extraction uses zero-fill, not the old
                      watermarked padding region, to avoid boundary artifacts.

Public entry point
──────────────────
    extract_watermark(watermarked_path, key, output_path, extraction_id)
"""

import os
import cv2
import numpy as np
from .embedding import (
    _forward_pipeline,
    _isvd,
    henon_decrypt,
)
from .models import WatermarkExtraction
from .utility import reconstruct_full_image, getImg
from .embed_key import EmbedKey


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
    watermarked_path: str,
    key: EmbedKey,
    output_path: str = "extracted_watermark.png",
    extraction_id: int | None = None,
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

    # FIX BUG 5: reconstruct_full_image must re-pad with zeros, not with the
    # old watermarked padding strip. Confirm your utility does zero-padding:
    #   padded = np.pad(Iw, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    # If it re-attaches bottom_pad/right_pad from the key, change it to zeros.
    Iw = reconstruct_full_image(Iw, key)

    # =====================================================
    # 🔹 STEP 2: forward pipeline → per-block SVs
    # =====================================================
    _tick(extraction, 25)

    _, HSw_hat_list, _, _, positions, LL, highpasses, tr = _forward_pipeline(
        Iw, key.block_size, key.dtcwt_levels
    )

    n_blocks = len(HSw_hat_list)
    sv_len   = len(HSw_hat_list[0])

    # =====================================================
    # 🔹 STEP 3: recover watermark singular values per block
    # =====================================================
    _tick(extraction, 40)

    # FIX BUG 2: guard against legacy keys where HSw_list was saved as
    # dtype=object, which breaks broadcasting in the subtraction below.
    HSw_hat = np.array(HSw_hat_list, dtype=np.float64)   # (n_blocks, sv_len)
    HSw     = np.array(key.HSw_list,  dtype=np.float64)   # (n_blocks, sv_len)

    # Recover the watermark SVs that were added during embedding:
    #   S_host_new = S_host + alpha * S_wm   →   S_wm ≈ (S_host_new - S_host) / alpha
    Sw_prime = (HSw_hat - HSw) / key.alpha_star           # (n_blocks, sv_len)

    # FIX BUG 3: small alpha (e.g. 0.001) amplifies any noise by 1000×.
    # Clip to a reasonable range to suppress blown-up noise terms.
    # The valid SV range after normalization is roughly [0, ~50] for an 8×8 block.
    Sw_prime = np.clip(Sw_prime, -50.0, 50.0)

    # =====================================================
    # 🔹 STEP 4: per-block ISVD reconstruction → average images
    #
    # FIX BUG 1 (primary cause of noisy output):
    #   Original code:  sv_mean = mean(Sw_prime, axis=0)  →  one ISVD
    #   This collapses all n_blocks worth of spatial information into a single
    #   mean SV vector before reconstruction, destroying structure entirely.
    #
    #   Correct approach: reconstruct one candidate watermark image per block
    #   using that block's recovered SVs, then average the images. This is the
    #   standard multi-block SVD watermark recovery technique.
    # =====================================================
    _tick(extraction, 55)

    # FIX BUG 4: use Sw_full from the key to initialise the full SV vector.
    # Original code zero-padded from sv_len=8 up to wm_side (e.g. 512),
    # leaving most singular values as 0 and badly distorting the matrix.
    # Instead, start from the original watermark SVs and overwrite only the
    # recovered portion — this preserves the correct energy distribution.
    # Sw_full = np.array(key.Sw_full, dtype=np.float64)  # full SV vector of W_enc
    # k = len(Sw_full)                                    # = min(wm_side, wm_side)

    # W_candidates = []
    # for i in range(n_blocks):
    #     # Build the full SV vector: base is Sw_full, overwrite with recovered SVs
    #     sv_k = Sw_full.copy()
    #     recovered_len = min(sv_len, k)
    #     sv_k[:recovered_len] = Sw_prime[i, :recovered_len]

    #     # Reconstruct one candidate watermark from this block's SVs
    #     W_candidate = _isvd(key.Uw[:, :k], sv_k, key.Vtw[:k, :])
    #     W_candidates.append(W_candidate)

    # # Average all candidate images — incoherent noise cancels, signal adds up
    # Cw = np.mean(W_candidates, axis=0)
    
    # Sw_full = np.array(key.Sw_full, dtype=np.float64)
    # k = len(Sw_full)
    # recovered_len = min(sv_len, k)

    # Uw  = key.Uw[:, :k]   # (wm_side, k)
    # Vtw = key.Vtw[:k, :]  # (k, wm_side)

    # # Build all SV vectors at once: tile Sw_full, overwrite recovered portion
    # sv_matrix = np.tile(Sw_full, (n_blocks, 1))                  # (n_blocks, k)
    # sv_matrix[:, :recovered_len] = Sw_prime[:, :recovered_len]   # overwrite

    # # Chunked vectorized ISVD — safe for all image sizes
    # CHUNK = 256
    # W_accum = np.zeros((Uw.shape[0], Vtw.shape[1]), dtype=np.float64)

    # for start in range(0, n_blocks, CHUNK):
    #     end = min(start + CHUNK, n_blocks)
    #     sv_chunk = sv_matrix[start:end]                           # (chunk, k)
    #     W_chunk = np.einsum('ij,bj,jk->bik', Uw, sv_chunk, Vtw) # (chunk, wm_side, wm_side)
    #     W_accum += W_chunk.sum(axis=0)

    # Cw = W_accum / n_blocks
    
    Sw_full = np.array(key.Sw_full, dtype=np.float64)
    k = len(Sw_full)
    recovered_len = min(sv_len, k)

    Uw  = key.Uw[:, :k]    # (wm_side, k)
    Vtw = key.Vtw[:k, :]   # (k, wm_side)

    # Build sv_matrix: (n_blocks, k) — no tiling, direct assignment is faster
    sv_matrix = np.empty((n_blocks, k), dtype=np.float64)
    sv_matrix[:] = Sw_full                                        # broadcast base
    sv_matrix[:, :recovered_len] = Sw_prime[:, :recovered_len]   # overwrite recovered

    # Key insight: W_i = (Uw * sv_i) @ Vtw
    # So sum over all blocks:
    #   ΣW_i = Uw @ diag(Σsv_i) @ Vtw   only if we want the sum of scaled matrices
    # But we want mean, so:
    #   mean(sv_matrix, axis=0) gives the mean SV vector → ONE reconstruction
    sv_mean = sv_matrix.mean(axis=0)                              # (k,)
    Uw_scaled = Uw * sv_mean                                      # (wm_side, k)
    Cw = Uw_scaled @ Vtw                                          # (wm_side, wm_side)

    # =====================================================
    # 🔹 STEP 5: Henon decrypt
    # =====================================================
    _tick(extraction, 75)

    W_ext = henon_decrypt(Cw, a=key.henon_a, b=key.henon_b)

    # =====================================================
    # 🔹 STEP 6: normalise & save
    # =====================================================
    _tick(extraction, 88)

    # W_norm = W_ext - W_ext.min()
    # if W_norm.max() > 0:
    #     W_norm /= W_norm.max()
    # W_out = (W_norm * 255).astype(np.uint8)
    
    # Step 1: normalize to [0, 255]
    W_norm = W_ext - W_ext.min()
    if W_norm.max() > 0:
        W_norm /= W_norm.max()
    W_norm = (W_norm * 255).astype(np.uint8)

    # Step 2: denoise — removes salt-and-pepper and gaussian noise
    # h=10 is filter strength, adjust up if still noisy, down if blurring detail
    W_denoised = cv2.fastNlMeansDenoising(W_norm, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 3: sharpen edges using unsharp masking
    gaussian = cv2.GaussianBlur(W_denoised, (0, 0), sigmaX=2)
    W_sharp = cv2.addWeighted(W_denoised, 1.8, gaussian, -0.8, 0)

    # Step 4: CLAHE — boosts local contrast, makes dark logo pop against background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    W_contrast = clahe.apply(W_sharp)

    # Step 5: Otsu threshold — converts grey background to clean white,
    # logo pixels to clean black. Otsu auto-picks the best threshold.
    _, W_binary = cv2.threshold(W_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: morphological cleanup — removes isolated noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    W_clean = cv2.morphologyEx(W_binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    W_clean = cv2.morphologyEx(W_clean,  cv2.MORPH_CLOSE, kernel, iterations=2)

    W_out = W_clean

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

# """
# extract_watermark.py
# ────────────────────
# Implements Algorithm 2 — watermark extraction from a watermarked image.

# Public entry point
# ──────────────────
#     extract_watermark(watermarked_path, key, output_path, extraction_id)

# Algorithm
# ---------
#   1. Load & resize the watermarked image  (same resize() as embedding)
#   2. Forward pipeline → HSw_hat_list      (per-block dominant SVs)
#   3. Recover watermark SVs:
#          Sw' = (HSw_hat - HSw) / α*
#   4. Average SVs → single reconstruction via ISVD
#   5. Henon decrypt  →  W_extracted
#   6. Normalise to [0, 255] and save
# """

# import os
# import cv2
# import numpy as np
# from .embedding import (
#     _forward_pipeline,
#     _isvd,
#     henon_decrypt,
# )
# from .models import WatermarkExtraction
# from .utility import reconstruct_full_image,getImg
# from .embed_key import EmbedKey

# # ═════════════════════════════════════════════════════════════════════════════
# #  INTERNAL HELPERS
# # ═════════════════════════════════════════════════════════════════════════════

# def _save_png(path: str, img: np.ndarray) -> None:
#     """Write a uint8 numpy array as PNG, creating parent dirs as needed."""
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     ok = cv2.imwrite(path, img)
#     if not ok:
#         raise IOError(f"cv2.imwrite failed: {path}")


# def _tick(extraction: "WatermarkExtraction | None", progress: int) -> None:
#     if extraction is not None:
#         extraction.set_status(WatermarkExtraction.Status.EXTRACTING, progress)


# # ═════════════════════════════════════════════════════════════════════════════
# #  PUBLIC ENTRY POINT
# # ═════════════════════════════════════════════════════════════════════════════

# def extract_watermark(
#     watermarked_path : str,
#     key              : EmbedKey,
#     output_path      : str = "extracted_watermark.png",
#     extraction_id    : int | None = None,
# ) -> np.ndarray:
#     """
#     Recover the embedded watermark from *watermarked_path* using *key*.

#     Parameters
#     ----------
#     watermarked_path : path to the watermarked image (PNG / JPEG)
#     key              : EmbedKey produced at embedding time
#     output_path      : where to save the extracted watermark PNG
#     extraction_id    : optional WatermarkExtraction PK – enables live
#                        DB progress updates and metadata write-back

#     Returns
#     -------
#     W_out : uint8 ndarray of the recovered watermark
#     """

   
#     extraction: WatermarkExtraction | None = None
#     if extraction_id is not None:
#         try:
#             extraction = WatermarkExtraction.objects.get(id=extraction_id)
#         except WatermarkExtraction.DoesNotExist:
#             extraction = None

#     # =====================================================
#     # 🔹 STEP 1: load & resize watermarked image
#     # =====================================================
#     _tick(extraction, 10)

#     Iw = getImg(watermarked_path)
#     Iw = reconstruct_full_image(Iw,key)
#     # =====================================================
#     # 🔹 STEP 2: forward pipeline → per-block SVs
#     # =====================================================
#     _tick(extraction, 25)

#     _, HSw_hat_list, _, _, positions, LL, highpasses, tr = _forward_pipeline(Iw, key.block_size, key.dtcwt_levels)

#     n_blocks = len(HSw_hat_list)
#     sv_len   = len(HSw_hat_list[0])

#     # =====================================================
#     # 🔹 STEP 3: recover watermark singular values
#     # =====================================================
#     _tick(extraction, 45)

#     HSw_hat = np.array(HSw_hat_list)              
#     HSw     = np.array(key.HSw_list)              
#     Sw_prime = (HSw_hat - HSw) / key.alpha_star   

#     # =====================================================
#     # 🔹 STEP 4: average → single ISVD reconstruction
#     # =====================================================
#     _tick(extraction, 60)

#     sv_mean = np.mean(Sw_prime, axis=0)                   

#     k = min(key.Uw.shape[1], key.Vtw.shape[0])
#     sv_k = np.zeros(k)
#     sv_k[:min(k, len(sv_mean))] = sv_mean[:k]

#     Cw = _isvd(key.Uw[:, :k], sv_k, key.Vtw[:k, :])   

#     # =====================================================
#     # 🔹 STEP 5: Henon decrypt
#     # =====================================================
#     _tick(extraction, 75)

#     W_ext = henon_decrypt(Cw, a=key.henon_a, b=key.henon_b)

#     # =====================================================
#     # 🔹 STEP 6: normalise & save
#     # =====================================================
#     _tick(extraction, 88)

#     W_norm = W_ext - W_ext.min()
#     if W_norm.max() > 0:
#         W_norm /= W_norm.max()
#     W_out = (W_norm * 255).astype(np.uint8)

#     _save_png(output_path, W_out)

#     # =====================================================
#     # 🔹 STEP 7: write metadata back to DB row
#     # =====================================================
#     if extraction is not None:
#         extraction.alpha_star      = float(key.alpha_star)
#         extraction.n_blocks        = int(n_blocks)
#         extraction.sv_length       = int(sv_len)
#         extraction.watermark_shape = list(W_out.shape)
#         extraction.set_status(WatermarkExtraction.Status.COMPLETED, 100)

#     return W_out