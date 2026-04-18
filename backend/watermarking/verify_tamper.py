"""
verify_tamper.py
────────────────
Standalone tamper-detection module.

Public entry point
──────────────────
    verify_tamper(received_path, key, tamper_map_path, overlay_path)

It mirrors the style of embed_watermark() in embedding.py:
  • uses the same _forward_pipeline / resize helpers
  • saves images with cv2 (same as save_image)
  • updates a TamperVerification DB row at each stage
  • raises on hard errors so verify_runner.py can call mark_failed()
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .embedding import _forward_pipeline
from .models import TamperVerification
from .utility import reconstruct_full_image,getImg
from .embed_key import EmbedKey
from .path_helpers import path_verify_tamper_map, path_verify_overlay

# ═════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _save_png(path: str, img_bgr: np.ndarray) -> None:
    """Write a numpy array as PNG, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img_bgr)
    if not ok:
        raise IOError(f"cv2.imwrite failed for path: {path}")


def _build_tamper_map(
    tamper_grid: np.ndarray,  
    cell_px: int,
    out_w: int,
    out_h: int,
) -> Image.Image:
    """
    Produce a green/red block image at the natural DTCWT resolution
    then resize to (out_w, out_h) with nearest-neighbour.

    Green  (34, 197, 94)  → authentic block
    Red   (239,  68, 68)  → tampered  block
    """
    n_rows, n_cols = tamper_grid.shape
    map_w = n_cols * cell_px
    map_h = n_rows * cell_px

    img = Image.new("RGB", (map_w, map_h), (34, 197, 94))
    draw = ImageDraw.Draw(img)
    for row in range(n_rows):
        for col in range(n_cols):
            if tamper_grid[row, col]:
                x0 = col * cell_px
                y0 = row * cell_px
                draw.rectangle(
                    [x0, y0, x0 + cell_px - 1, y0 + cell_px - 1],
                    fill=(239, 68, 68),
                )

    return img.resize((out_w, out_h), Image.NEAREST)


def _build_overlay(
    received_bgr: np.ndarray,  # original-size BGR
    tamper_grid: np.ndarray,   # (n_rows, n_cols) bool
    cell_px: int,
    target_size: int,          # key.M – square target
) -> np.ndarray:
    """
    Resize received image to (target_size × target_size), then paint
    semi-transparent red rectangles over tampered blocks.
    Returns a BGR uint8 array.
    """
    n_rows, n_cols = tamper_grid.shape
    map_w = n_cols * cell_px
    map_h = n_rows * cell_px

    recv_resized = cv2.resize(
        received_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4
    )

    # Work in RGBA PIL for alpha-composite
    base = Image.fromarray(
        cv2.cvtColor(recv_resized, cv2.COLOR_BGR2RGBA)
    )
    overlay = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for row in range(n_rows):
        for col in range(n_cols):
            if tamper_grid[row, col]:
                x0 = int(col * cell_px * target_size / map_w)
                y0 = int(row * cell_px * target_size / map_h)
                x1 = int((col + 1) * cell_px * target_size / map_w) - 1
                y1 = int((row + 1) * cell_px * target_size / map_h) - 1
                # filled semi-transparent red
                draw.rectangle([x0, y0, x1, y1], fill=(239, 68, 68, 140))
                # solid red border
                draw.rectangle(
                    [x0, y0, x1, y1], outline=(220, 38, 38, 255), width=2
                )

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


def _print_summary(
    n_used: int,
    sv_deltas: np.ndarray,
    T: float,
    n_tampered: int,
    tampered_frac: float,
    is_tampered: bool,
) -> None:
    print(f"Blocks analysed : {n_used}")
    print(
        f"Max SV delta : {sv_deltas.max():.4f}  "
        f"(threshold = {T:.4f})"
    )
    print(
        f"Tampered blocks : {n_tampered}  "
        f"({tampered_frac * 100:.1f} %)"
    )
    verdict = "⚠ TAMPERED" if is_tampered else "✓ AUTHENTIC"
    print(f"Verdict → {verdict}")


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def verify_tamper(
    received_path: str,
    key: EmbedKey,
    reconstructed_watermark : str,
    tamper_map_path: str = "tamper_map.png",
    overlay_path: str = "tamper_overlay.png",
    verification_id: int | None = None,
) -> dict:
    """
    Detect and localise tampering in *received_path* using the stored
    dominant singular values in *key*.

    Algorithm
    ---------
    For every LL3 block i:
        delta_i = |received_dominant_SV_i  –  key.HSw_new_dominant[i]|
        delta_i <=  T  →  authentic
        delta_i  >  T  →  tampered

    The threshold T was calibrated during embedding against the
    observed benign PNG round-trip drift, so authentic images are
    immune to false positives caused by lossless re-compression.

    Parameters
    ----------
    received_path   : path to the image to examine (PNG/JPEG)
    key             : EmbedKey produced at embedding time
    tamper_map_path : where to write the green/red block map
    overlay_path    : where to write the received image with red overlay
    verification_id : optional TamperVerification PK – when supplied the
                      function updates the DB row with progress ticks

    Returns
    -------
    {
        "is_tampered"     : bool,
        "tampered_frac"   : float,          # 0.0 – 1.0
        "tamper_grid"     : np.ndarray,     # (n_rows, n_cols) bool
        "sv_deltas"       : np.ndarray,     # (n_rows, n_cols) float64
        "tamper_map_path" : str,
        "overlay_path"    : str,
    }
    """

    verification: TamperVerification | None = None
    if verification_id is not None:
        try:
            verification = TamperVerification.objects.get(id=verification_id)
        except TamperVerification.DoesNotExist:
            verification = None

    def _tick(progress: int) -> None:
        if verification is not None:
            verification.set_status(TamperVerification.Status.VERIFYING, progress)

    # =====================================================
    # 🔹 STEP 1: load & resize received image
    # =====================================================
    _tick(10)

    img_recv = getImg(received_path)  
    img_recv = reconstruct_full_image(img_recv,key)
    
    # _save_png(reconstructed_watermark, img_recv)
    
    # =====================================================
    # 🔹 STEP 2: forward pipeline
    # =====================================================
    _tick(25)

    (_, HSw_hat_list, _, _, positions,
     LL, highpasses, tr) = _forward_pipeline(
        img_recv, key.block_size, key.dtcwt_levels
    )

    LL_shape = LL.shape
    n_blocks = len(HSw_hat_list)
    T        = key.tamper_threshold
    print(f"           Threshold T = {T:.4f}  |  blocks = {n_blocks}")

    # =====================================================
    # 🔹 STEP 3: per-block SV distance
    # =====================================================
    _tick(45)

    sv_deltas_flat = np.array(
        [abs(HSw_hat_list[i][0] - key.HSw_new_dominant[i])
         for i in range(n_blocks)],
        dtype=np.float64,
    )
    tamper_flat = sv_deltas_flat > T

    n_rows  = LL_shape[0] // key.block_size
    n_cols  = LL_shape[1] // key.block_size
    n_used  = n_rows * n_cols

    tamper_grid    = tamper_flat[:n_used].reshape(n_rows, n_cols)
    sv_deltas_grid = sv_deltas_flat[:n_used].reshape(n_rows, n_cols)

    n_tampered    = int(tamper_flat[:n_used].sum())
    tampered_frac = n_tampered / max(n_used, 1)
    is_tampered   = bool(n_tampered > 0)

    _print_summary(n_used, sv_deltas_flat[:n_used], T,
                   n_tampered, tampered_frac, is_tampered)

    # =====================================================
    # 🔹 STEP 4: build visual outputs
    # =====================================================
    _tick(65)

    scale   = 2 ** key.dtcwt_levels
    cell_px = key.block_size * scale

    tmap_pil = _build_tamper_map(tamper_grid, cell_px, key.M, key.M)
    tmap_bgr = cv2.cvtColor(np.array(tmap_pil), cv2.COLOR_RGB2BGR)
    _save_png(tamper_map_path, tmap_bgr)
    print(f"           Tamper map  → {tamper_map_path}")

    received_bgr = cv2.imread(received_path, cv2.IMREAD_COLOR)
    if received_bgr is None:
        raise ValueError(f"Could not open received image: {received_path}")

    overlay_bgr = _build_overlay(received_bgr, tamper_grid, cell_px, key.M)
    _save_png(overlay_path, overlay_bgr)
    print(f"           Overlay     → {overlay_path}")

    _tick(90)

    # =====================================================
    # 🔹 STEP 5: persist results to DB row
    # =====================================================
    if verification is not None:
        verification.is_tampered          = is_tampered
        verification.tampered_frac        = tampered_frac
        verification.tamper_grid_flat     = tamper_grid.astype(int).flatten().tolist()
        verification.grid_rows            = int(n_rows)
        verification.grid_cols            = int(n_cols)
        verification.sv_deltas_flat       = sv_deltas_grid.flatten().tolist()
        verification.tamper_threshold_used = float(T)
        verification.set_status(TamperVerification.Status.COMPLETED, 100)
        print("[verify_tamper]  DB row updated → COMPLETED")

    return {
        "is_tampered"     : is_tampered,
        "tampered_frac"   : tampered_frac,
        "tamper_grid"     : tamper_grid,
        "sv_deltas"       : sv_deltas_grid,
        "tamper_map_path" : tamper_map_path,
        "overlay_path"    : overlay_path,
    }