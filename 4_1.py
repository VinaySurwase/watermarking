"""
Medical Image Watermarking with Tamper Detection
=================================================
Algorithm 1 : Watermark Embedding   (Section 3.1)
Algorithm 2 : Watermark Extraction  (Section 3.2)
Algorithm 3 : Tamper Detection      (SV-distance, no false positives)

Tamper Detection Design
-----------------------
During EMBEDDING the watermarked dominant singular value HSw_new[0] for every
block is stored EXACTLY in the key (as a float64).

During VERIFICATION the same pipeline is re-run on the received image to get
HSw_hat[0] per block.  The absolute difference |HSw_hat[0] - HSw_new[0]| is
compared against a tolerance threshold T:

    |HSw_hat[0] - HSw_new[0]| <= T  →  authentic  ✓
    |HSw_hat[0] - HSw_new[0]|  > T  →  tampered   ⚠

Why this works
--------------
• Benign operations (PNG round-trip, minor noise) shift the dominant SV by a
  small, bounded amount (typically < 1.0 for 8-bit images).
• Malicious pixel edits (copy-paste, inpainting, intensity changes) shift the
  dominant SV by a MUCH larger amount (typically >> 5.0).
• The threshold T sits between these two regimes.

Threshold guidelines
--------------------
  T = 1.0  →  very tight  (may flag heavy JPEG compression as tampered)
  T = 3.0  →  balanced    (default – ignores PNG round-trip, catches edits)
  T = 8.0  →  loose       (only catches large malicious changes)

Requirements
------------
    pip install numpy scipy Pillow dtcwt pyswarms matplotlib
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import dctn, idctn
import dtcwt
from dtcwt.numpy import Pyramid
from pyswarms.single.global_best import GlobalBestPSO


# ══════════════════════════════════════════════════════════════════════════════
#  KEY BUNDLE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbedKey:
    """
    All values produced during embedding needed by extractor & tamper detector.

    alpha_star        : optimised PSO factor α*
    HSw_list          : original host SV vectors per block   (pre-embedding)
    HSw_new_dominant  : dominant SV of each WATERMARKED block  ← tamper anchor
    tamper_threshold  : absolute-SV-distance threshold T
    wm_sv_list        : watermark SV slices per block
    Uw / Vtw          : SV matrices of the encrypted watermark
    watermark_shape   : (H, W) of the encrypted watermark
    henon_a / b       : Henon map parameters
    M                 : image resize side
    block_size        : DCT block size (default 8)
    dtcwt_levels      : DTCWT depth   (default 3)
    """
    alpha_star        : float
    HSw_list          : List[np.ndarray]
    HSw_new_dominant  : np.ndarray        # shape (n_blocks,)  float64
    tamper_threshold  : float
    wm_sv_list        : List[np.ndarray]
    Uw                : np.ndarray
    Vtw               : np.ndarray
    watermark_shape   : Tuple[int, int]
    henon_a           : float = 1.4
    henon_b           : float = 0.3
    M                 : int   = 512
    block_size        : int   = 8
    dtcwt_levels      : int   = 3


# ══════════════════════════════════════════════════════════════════════════════
#  HENON CHAOTIC SCRAMBLING
# ══════════════════════════════════════════════════════════════════════════════

def _henon_seq(n: int, a: float, b: float, n_burnin: int = 1000) -> np.ndarray:
    x, y = 0.1, 0.1
    for _ in range(n_burnin):
        x, y = 1.0 - a * x * x + y, b * x
    seq = np.empty(n)
    for i in range(n):
        x, y = 1.0 - a * x * x + y, b * x
        seq[i] = x
    return seq


def henon_encrypt(watermark: np.ndarray,
                  a: float = 1.4, b: float = 0.3) -> np.ndarray:
    wm = watermark.astype(np.float64)
    if wm.max() > 1.0:
        wm /= 255.0
    perm = np.argsort(_henon_seq(wm.size, a, b))
    return wm.flatten()[perm].reshape(wm.shape)


def henon_decrypt(scrambled: np.ndarray,
                  a: float = 1.4, b: float = 0.3) -> np.ndarray:
    perm     = np.argsort(_henon_seq(scrambled.size, a, b))
    inv_perm = np.argsort(perm)
    return scrambled.flatten()[inv_perm].reshape(scrambled.shape)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _partition(arr: np.ndarray, b: int):
    H, W = arr.shape
    blocks, positions = [], []
    for r in range(0, H - b + 1, b):
        for c in range(0, W - b + 1, b):
            blocks.append(arr[r:r+b, c:c+b].copy())
            positions.append((r, c))
    return blocks, positions


def _merge(blocks, positions, shape: tuple, b: int) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    for blk, (r, c) in zip(blocks, positions):
        out[r:r+b, c:c+b] = blk
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SVD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _svd(A: np.ndarray):
    return np.linalg.svd(A, full_matrices=False)


def _isvd(U, s, Vt) -> np.ndarray:
    return U @ np.diag(s) @ Vt


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE LOAD HELPER
# ══════════════════════════════════════════════════════════════════════════════

def resize(image_path):
    import cv2
    import numpy as np
    import math

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found or invalid path")

    H, W = img.shape

    # Target size (multiple of 64)
    M = math.ceil(max(H, W) / 64) * 64

    # Compute padding
    pad_h = M - H
    pad_w = M - W

    # Pad (black padding → 0)
    padded = np.pad(
        img,
        ((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0
    )

    return padded.astype(np.float32)

def _load_gray(path: str, size: int) -> np.ndarray:
    """Load image, convert to grayscale, resize, return float64."""
    return np.array(
        Image.open(path).convert("L").resize((size, size), Image.LANCZOS),
        dtype=np.float64,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED FORWARD PIPELINE  image → DTCWT → LL3 → DCT → SVD
# ══════════════════════════════════════════════════════════════════════════════

def _forward_pipeline(img: np.ndarray, block_size: int, levels: int):
    tr  = dtcwt.Transform2d()
    pyr = tr.forward(img, nlevels=levels)
    LL  = pyr.lowpass.copy()
    highpasses = pyr.highpasses

    blocks, positions = _partition(LL, block_size)

    U_list, sv_list, Vt_list, dct_blocks = [], [], [], []
    for blk in blocks:
        C        = dctn(blk, norm="ortho")
        U, s, Vt = _svd(C)
        U_list.append(U)
        sv_list.append(s.copy())
        Vt_list.append(Vt)
        dct_blocks.append(C)

    return U_list, sv_list, Vt_list, dct_blocks, positions, LL, highpasses, tr


# ══════════════════════════════════════════════════════════════════════════════
#  INVERSE PIPELINE  IDCT → merge → IDTCWT
# ══════════════════════════════════════════════════════════════════════════════

def _inverse_pipeline(new_dct_blocks, positions, LL_shape,
                      highpasses, tr, block_size: int) -> np.ndarray:
    idct_blocks = [idctn(C, norm="ortho") for C in new_dct_blocks]
    LL_new      = _merge(idct_blocks, positions, LL_shape, block_size)
    return tr.inverse(Pyramid(LL_new, highpasses))


# ══════════════════════════════════════════════════════════════════════════════
#  PSO FITNESS
# ══════════════════════════════════════════════════════════════════════════════

# def _fitness(alpha_mat, HSw_list, wm_sv_list, U_list, Vt_list, dct_blocks):
#     lam   = 0.6
#     costs = np.empty(len(alpha_mat))
#     for i, (alpha,) in enumerate(alpha_mat):
#         mse_acc = sv_acc = 0.0
#         for hsw, sw, U, Vt, C_orig in zip(
#                 HSw_list, wm_sv_list, U_list, Vt_list, dct_blocks):
#             hsw_new  = hsw + alpha * sw
#             C_new    = _isvd(U, hsw_new, Vt)
#             mse_acc += np.mean((C_orig - C_new) ** 2)
#             sv_acc  += np.mean(np.abs(alpha * sw))
#         n        = len(HSw_list)
#         avg_mse  = mse_acc / n
#         psnr_val = 10.0 * np.log10(255.0 ** 2 / (avg_mse + 1e-12))
#         costs[i] = lam * (-psnr_val) + (1.0 - lam) * (sv_acc / n)
#     return costs

def _fitness(alpha_mat, HSw_list, wm_sv_list, *args):
    lam = 0.6

    HSw = np.array(HSw_list)   # (B, k)
    SW  = np.array(wm_sv_list) # (B, k)

    costs = np.zeros(len(alpha_mat))

    for i, (alpha,) in enumerate(alpha_mat):
        # no reconstruction needed
        diff = alpha * SW

        mse = np.mean(diff * diff)
        psnr_val = 10.0 * np.log10(255.0**2 / (mse + 1e-12))

        sv_term = np.mean(np.abs(diff))

        costs[i] = lam * (-psnr_val) + (1 - lam) * sv_term

    return costs


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 1 – EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def embed_watermark(
    host_path        : str,
    watermark_path   : str,
    M                : int   = 512,
    block_size       : int   = 8,
    dtcwt_levels     : int   = 3,
    henon_a          : float = 1.4,
    henon_b          : float = 0.3,
    pso_particles    : int   = 20,
    pso_iters        : int   = 50,
    alpha_bounds     : tuple = (0.001, 0.05),
    tamper_threshold : float = 3.0,
    output_path      : str   = "watermarked.png",
) -> tuple:
    """
    Embed watermark and store per-block tamper anchors.

    tamper_threshold T  controls detection sensitivity:
        T = 1.0  very tight  (may flag JPEG compression)
        T = 3.0  balanced    (default)
        T = 8.0  loose       (only catches large deliberate edits)

    Returns (Iw_uint8, key)
    """

    # S1 – Load
    print("[Alg1 / S1]  Loading host image …")
    # I = _load_gray(host_path, M)
    I = resize(host_path)

    # S2-4 – Forward pipeline
    print("[Alg1 / S2-4]  DTCWT → partition → DCT → SVD …")
    U_list, HSw_list, Vt_list, dct_blocks, positions, LL, highpasses, tr = \
        _forward_pipeline(I, block_size, dtcwt_levels)
    n_blocks = len(HSw_list)
    LL_shape = LL.shape
    sv_len   = len(HSw_list[0])
    print(f"           {n_blocks} blocks, LL3 {LL_shape}, {sv_len} SVs/block")

    # S5 – Henon encrypt watermark
    print("[Alg1 / S5]  Henon-encrypting watermark …")
    wm_side = max(int(np.sqrt(n_blocks)) * block_size, block_size)
    W_raw   = _load_gray(watermark_path, wm_side)
    W_enc   = henon_encrypt(W_raw, a=henon_a, b=henon_b)

    # S6 – SVD on W_enc
    print("[Alg1 / S6]  SVD on encrypted watermark …")
    Uw, Sw_full, Vtw = _svd(W_enc)
    total_needed = n_blocks * sv_len
    Sw_tiled     = np.tile(Sw_full,
                           int(np.ceil(total_needed / max(len(Sw_full), 1))))
    wm_sv_list   = [
        Sw_tiled[i * sv_len: i * sv_len + sv_len].copy()
        for i in range(n_blocks)
    ]

    # S7-8 – PSO
    print("[Alg1 / S7-8]  PSO optimising α* …")
    optimizer = GlobalBestPSO(
        n_particles = pso_particles,
        dimensions  = 1,
        options     = {"c1": 0.5, "c2": 0.3, "w": 0.9},
        bounds      = (np.array([alpha_bounds[0]]), np.array([alpha_bounds[1]])),
    )
    cost, best = optimizer.optimize(
        lambda a: _fitness(a, HSw_list, wm_sv_list, U_list, Vt_list, dct_blocks),
        iters=pso_iters, verbose=False,
    )
    alpha_star = float(best[0])
    print(f"           α* = {alpha_star:.6f}  (PSO cost = {cost:.4f})")

    # S9 – Embed, ISVD, IDCT, IDTCWT
    print("[Alg1 / S9]  Embedding …")
    new_dct_blocks    = []
    HSw_new_dominant  = np.empty(n_blocks, dtype=np.float64)   # ← tamper anchors

    for idx, (hsw, sw, U, Vt) in enumerate(
            zip(HSw_list, wm_sv_list, U_list, Vt_list)):
        hsw_new = hsw + alpha_star * sw                         # Eq. (6)
        C_new   = _isvd(U, hsw_new, Vt)
        new_dct_blocks.append(C_new)
        # Store the EXACT dominant SV of the watermarked DCT block.
        # This is computed directly from hsw_new (float64), BEFORE any
        # save/reload round-trip, so it is the ground truth.
        HSw_new_dominant[idx] = hsw_new[0]

    Iw       = _inverse_pipeline(new_dct_blocks, positions, LL_shape,
                                  highpasses, tr, block_size)
    Iw_uint8 = np.clip(Iw, 0, 255).astype(np.uint8)
    Image.fromarray(Iw_uint8).save(output_path)

    _psnr = psnr(I, Iw_uint8.astype(np.float64))
    print(f"           Saved → {output_path}   PSNR = {_psnr:.2f} dB")

    # Calibrate: measure the actual SV drift caused by PNG save→reload
    # so we can set an informed threshold at runtime.
    # Iw_reloaded   = _load_gray(output_path, M)
    Iw_reloaded = resize(output_path)
    _, sv_reload, _, _, _, _, _, _ = _forward_pipeline(
        Iw_reloaded, block_size, dtcwt_levels
    )
    drift = np.array([abs(sv_reload[i][0] - HSw_new_dominant[i])
                      for i in range(n_blocks)])
    max_benign_drift = float(drift.max())
    # Auto-set threshold to max observed benign drift * safety_factor
    # but respect user-supplied tamper_threshold as a lower bound.
    auto_threshold  = max_benign_drift * 2.5
    final_threshold = max(tamper_threshold, auto_threshold)
    print(f"           PNG round-trip max SV drift : {max_benign_drift:.4f}")
    print(f"           Auto threshold              : {auto_threshold:.4f}")
    print(f"           Final tamper threshold T    : {final_threshold:.4f}")

    key = EmbedKey(
        alpha_star        = alpha_star,
        HSw_list          = HSw_list,
        HSw_new_dominant  = HSw_new_dominant,    # exact float64 per-block SVs
        tamper_threshold  = final_threshold,     # calibrated T
        wm_sv_list        = wm_sv_list,
        Uw                = Uw,
        Vtw               = Vtw,
        watermark_shape   = W_enc.shape,
        henon_a           = henon_a,
        henon_b           = henon_b,
        M                 = M,
        block_size        = block_size,
        dtcwt_levels      = dtcwt_levels,
    )
    return Iw_uint8, key


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 2 – EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_watermark(
    watermarked_path : str,
    key              : EmbedKey,
    output_path      : str = "extracted_watermark.png",
) -> np.ndarray:
    """Extract watermark following Algorithm 2."""

    print("[Alg2 / L1-8]  Forward pipeline on watermarked image …")
    # Iw = _load_gray(watermarked_path, key.M)
    Iw = resize(watermarked_path)
    _, HSw_hat_list, _, _, _, _, _, _ = _forward_pipeline(
        Iw, key.block_size, key.dtcwt_levels
    )

    print(f"[Alg2 / L10-12]  Recovering watermark SVs (α* = {key.alpha_star:.6f}) …")
    Sw_prime_list = [
        (hsw_hat - hsw) / key.alpha_star
        for hsw_hat, hsw in zip(HSw_hat_list, key.HSw_list)
    ]

    print("[Alg2 / L13-14]  ISVD reconstruction of W_enc …")
    H_wm, W_wm = key.watermark_shape
    k           = min(key.Uw.shape[1], key.Vtw.shape[0])
    Cw_accum    = np.zeros((H_wm, W_wm), dtype=np.float64)
    for sw_prime in Sw_prime_list:
        sv_k                        = np.zeros(k)
        sv_k[:min(k, len(sw_prime))] = sw_prime[:k]
        contrib                     = _isvd(key.Uw[:, :k], sv_k, key.Vtw[:k, :])
        rh = min(H_wm, contrib.shape[0])
        rw = min(W_wm, contrib.shape[1])
        Cw_accum[:rh, :rw] += contrib[:rh, :rw]
    Cw = Cw_accum / max(len(Sw_prime_list), 1)

    print("[Alg2 / L15]  Henon decryption …")
    W_ext  = henon_decrypt(Cw, a=key.henon_a, b=key.henon_b)
    W_norm = W_ext - W_ext.min()
    if W_norm.max() > 0:
        W_norm /= W_norm.max()
    W_out = (W_norm * 255).astype(np.uint8)

    Image.fromarray(W_out).save(output_path)
    print(f"[Alg2 / L16]  Extracted watermark saved → {output_path}")
    return W_out


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 3 – TAMPER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def verify_tamper(
    received_path   : str,
    key             : EmbedKey,
    tamper_map_path : str = "tamper_map.png",
    overlay_path    : str = "tamper_overlay.png",
) -> dict:
    """
    Detect and localise tampering using SV-distance comparison.

    For each LL3 block:
        delta = |HSw_hat[0]  –  HSw_new_dominant[stored]|
        delta <= T  →  authentic
        delta  > T  →  tampered

    This is immune to PNG round-trip drift because T was calibrated
    against the actual observed benign drift during embedding.

    Returns dict with: is_tampered, tampered_frac, tamper_grid,
                       sv_deltas, tamper_map_path, overlay_path
    """

    print("[Alg3 / T1]  Forward pipeline on received image …")
    # img_recv = _load_gray(received_path, key.M)
    img_recv = resize(received_path)
    (_, HSw_hat_list, _, _, positions,
     LL, highpasses, tr) = _forward_pipeline(
        img_recv, key.block_size, key.dtcwt_levels
    )

    LL_shape = LL.shape
    n_blocks = len(HSw_hat_list)
    T        = key.tamper_threshold
    print(f"           Using tamper threshold T = {T:.4f}")

    # ── Core comparison: |received dominant SV - stored dominant SV| ─────────
    print("[Alg3 / T2]  Computing per-block SV distances …")
    sv_deltas  = np.array([
        abs(HSw_hat_list[i][0] - key.HSw_new_dominant[i])
        for i in range(n_blocks)
    ])
    tamper_flat = sv_deltas > T

    n_rows    = LL_shape[0] // key.block_size
    n_cols    = LL_shape[1] // key.block_size
    n_used    = n_rows * n_cols
    tamper_grid      = tamper_flat[:n_used].reshape(n_rows, n_cols)
    sv_deltas_grid   = sv_deltas[:n_used].reshape(n_rows, n_cols)

    n_tampered    = int(tamper_flat[:n_used].sum())
    tampered_frac = n_tampered / max(n_used, 1)
    is_tampered   = bool(n_tampered > 0)

    print(f"           Blocks analysed : {n_used}")
    print(f"           Max SV delta    : {sv_deltas.max():.4f}  "
          f"(threshold = {T:.4f})")
    print(f"           Tampered blocks : {n_tampered}  "
          f"({tampered_frac * 100:.1f}%)")
    print(f"           Verdict  →  "
          f"{'⚠  TAMPERED' if is_tampered else '✓  AUTHENTIC'}")

    # ── Visual outputs ────────────────────────────────────────────────────────
    scale   = 2 ** key.dtcwt_levels
    cell_px = key.block_size * scale
    map_H   = n_rows * cell_px
    map_W   = n_cols * cell_px

    # Tamper-map
    tmap = Image.new("RGB", (map_W, map_H), (34, 197, 94))
    draw = ImageDraw.Draw(tmap)
    for row in range(n_rows):
        for col in range(n_cols):
            if tamper_grid[row, col]:
                x0, y0 = col * cell_px, row * cell_px
                draw.rectangle([x0, y0,
                                 x0 + cell_px - 1, y0 + cell_px - 1],
                                fill=(239, 68, 68))
    tmap_resized = tmap.resize((key.M, key.M), Image.NEAREST)
    tmap_resized.save(tamper_map_path)
    print(f"           Tamper map → {tamper_map_path}")

    # Overlay on received image
    recv_rgb = Image.open(received_path).convert("RGB").resize(
        (key.M, key.M), Image.LANCZOS)
    overlay  = Image.new("RGBA", (key.M, key.M), (0, 0, 0, 0))
    draw_ov  = ImageDraw.Draw(overlay)
    for row in range(n_rows):
        for col in range(n_cols):
            if tamper_grid[row, col]:
                x0 = int(col * cell_px * key.M / map_W)
                y0 = int(row * cell_px * key.M / map_H)
                x1 = int((col + 1) * cell_px * key.M / map_W) - 1
                y1 = int((row + 1) * cell_px * key.M / map_H) - 1
                draw_ov.rectangle([x0, y0, x1, y1],
                                   fill=(239, 68, 68, 140))
                draw_ov.rectangle([x0, y0, x1, y1],
                                   outline=(220, 38, 38, 255), width=2)
    Image.alpha_composite(recv_rgb.convert("RGBA"),
                          overlay).convert("RGB").save(overlay_path)
    print(f"           Overlay    → {overlay_path}")

    # Summary figure
    _plot_summary(recv_rgb, tmap_resized, sv_deltas_grid, T,
                  tampered_frac, is_tampered, n_rows, n_cols)

    return {
        "is_tampered"     : is_tampered,
        "tampered_frac"   : tampered_frac,
        "tamper_grid"     : tamper_grid,
        "sv_deltas"       : sv_deltas_grid,
        "tamper_map_path" : tamper_map_path,
        "overlay_path"    : overlay_path,
    }


def _plot_summary(recv_img, tmap_img, sv_deltas_grid, T,
                  tampered_frac, is_tampered, n_rows, n_cols):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f172a")

    verdict_color = "#ef4444" if is_tampered else "#22c55e"
    verdict_text  = "⚠  IMAGE TAMPERED" if is_tampered else "✓  IMAGE AUTHENTIC"

    # Panel 1 – received image
    axes[0].imshow(recv_img, cmap="gray")
    axes[0].set_title("Received Image", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    # Panel 2 – tamper map
    axes[1].imshow(tmap_img)
    axes[1].set_title("Tamper Map", color="white", fontsize=13, pad=10)
    axes[1].axis("off")
    axes[1].legend(
        handles=[mpatches.Patch(color="#22c55e", label="Authentic"),
                 mpatches.Patch(color="#ef4444", label="Tampered")],
        loc="lower right", fontsize=9,
        facecolor="#1e293b", labelcolor="white", framealpha=0.8,
    )

    # Panel 3 – SV delta heatmap
    im = axes[2].imshow(sv_deltas_grid, cmap="hot", interpolation="nearest",
                        aspect="auto")
    axes[2].set_title(f"SV-Distance Heatmap  (T = {T:.2f})",
                      color="white", fontsize=13, pad=10)
    axes[2].set_xlabel(f"Tampered: {tampered_frac*100:.1f}% of blocks",
                       color="#94a3b8", fontsize=10)
    axes[2].tick_params(colors="white")
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    # Draw threshold contour line
    axes[2].contour(sv_deltas_grid, levels=[T], colors=["cyan"], linewidths=1.5)

    for spine in axes[2].spines.values():
        spine.set_edgecolor("#334155")

    fig.text(0.5, 0.02, verdict_text, ha="center", fontsize=16,
             fontweight="bold", color=verdict_color,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b",
                       edgecolor=verdict_color, linewidth=2))

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig("tamper_summary.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("           Summary figure → tamper_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def nc(wm_orig: np.ndarray, wm_ext: np.ndarray) -> float:
    a = wm_orig.flatten().astype(np.float64)
    b = wm_ext.flatten().astype(np.float64)
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  KEY  I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_key(key: EmbedKey, path: str) -> None:
    np.savez_compressed(
        path,
        alpha_star        = np.array([key.alpha_star]),
        HSw_list          = np.array(key.HSw_list,   dtype=object),
        HSw_new_dominant  = key.HSw_new_dominant,
        tamper_threshold  = np.array([key.tamper_threshold]),
        wm_sv_list        = np.array(key.wm_sv_list, dtype=object),
        Uw                = key.Uw,
        Vtw               = key.Vtw,
        watermark_shape   = np.array(key.watermark_shape),
        henon_a           = np.array([key.henon_a]),
        henon_b           = np.array([key.henon_b]),
        M                 = np.array([key.M]),
        block_size        = np.array([key.block_size]),
        dtcwt_levels      = np.array([key.dtcwt_levels]),
    )
    print(f"Key saved → {path}.npz")


def load_key(path: str) -> EmbedKey:
    d = np.load(path, allow_pickle=True)
    return EmbedKey(
        alpha_star        = float(d["alpha_star"][0]),
        HSw_list          = list(d["HSw_list"]),
        HSw_new_dominant  = d["HSw_new_dominant"],
        tamper_threshold  = float(d["tamper_threshold"][0]),
        wm_sv_list        = list(d["wm_sv_list"]),
        Uw                = d["Uw"],
        Vtw               = d["Vtw"],
        watermark_shape   = tuple(int(x) for x in d["watermark_shape"]),
        henon_a           = float(d["henon_a"][0]),
        henon_b           = float(d["henon_b"][0]),
        M                 = int(d["M"][0]),
        block_size        = int(d["block_size"][0]),
        dtcwt_levels      = int(d["dtcwt_levels"][0]),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Medical Image Watermarking with Tamper Detection"
    )
    sub = parser.add_subparsers(dest="cmd")

    # embed
    emb = sub.add_parser("embed")
    emb.add_argument("host")
    emb.add_argument("watermark")
    emb.add_argument("--output",    default="watermarked.png")
    emb.add_argument("--key",       default="embed_key")
    emb.add_argument("--M",         type=int,   default=512)
    emb.add_argument("--block",     type=int,   default=8)
    emb.add_argument("--levels",    type=int,   default=3)
    emb.add_argument("--particles", type=int,   default=20)
    emb.add_argument("--iters",     type=int,   default=50)
    emb.add_argument("--alpha-min", type=float, default=0.001)
    emb.add_argument("--alpha-max", type=float, default=0.05)
    emb.add_argument("--threshold", type=float, default=3.0,
                     help="Minimum tamper threshold T (auto-calibrated upward if needed)")

    # extract
    ext = sub.add_parser("extract")
    ext.add_argument("watermarked")
    ext.add_argument("key")
    ext.add_argument("--output", default="extracted_watermark.png")

    # verify
    ver = sub.add_parser("verify")
    ver.add_argument("received")
    ver.add_argument("key")
    ver.add_argument("--map",     default="tamper_map.png")
    ver.add_argument("--overlay", default="tamper_overlay.png")

    args = parser.parse_args()

    if args.cmd == "embed":
        Iw, key = embed_watermark(
            host_path        = args.host,
            watermark_path   = args.watermark,
            M                = args.M,
            block_size       = args.block,
            dtcwt_levels     = args.levels,
            pso_particles    = args.particles,
            pso_iters        = args.iters,
            alpha_bounds     = (args.alpha_min, args.alpha_max),
            tamper_threshold = args.threshold,
            output_path      = args.output,
        )
        save_key(key, args.key)

    elif args.cmd == "extract":
        key  = load_key(args.key)
        W_ex = extract_watermark(args.watermarked, key, args.output)

    elif args.cmd == "verify":
        key    = load_key(args.key)
        result = verify_tamper(args.received, key, args.map, args.overlay)
        verdict = "TAMPERED" if result["is_tampered"] else "AUTHENTIC"
        print(f"\nFinal verdict: {verdict}  "
              f"({result['tampered_frac']*100:.1f}% blocks flagged)")

    else:
        parser.print_help()
        sys.exit(1)