"""
Medical Image Watermarking with Tamper Detection & Attack Classification
========================================================================
Algorithm 1 : Watermark Embedding    (Section 3.1)
Algorithm 2 : Watermark Extraction   (Section 3.2)
Algorithm 3 : Tamper Detection       (SV-distance, no false positives)
Algorithm 4 : Attack Classification  (Intentional vs Unintentional)

Tamper Detection Design
-----------------------
During EMBEDDING the watermarked dominant singular value HSw_new[0] for every
block is stored EXACTLY in the key (as a float64).

During VERIFICATION the absolute difference |HSw_hat[0] - HSw_new[0]| is
compared against a calibrated threshold T:

    delta <= T  →  authentic  ✓
    delta  > T  →  tampered   ⚠

Attack Classification Design  (Algorithm 4)
-------------------------------------------
Intentional and unintentional attacks leave fundamentally different fingerprints
in the SV-delta field.  Six discriminative features are extracted:

  F1  Spatial clustering (DBSCAN)
        Intentional → tampered blocks form tight connected clusters
        Unintentional → tampered blocks are scattered randomly

  F2  Delta magnitude ratio
        Intentional → large, sharp spikes  (delta >> T)
        Unintentional → small values just above T  (delta ≈ T)

  F3  Tampered area fraction
        Intentional → localised  (<30 % of image typically)
        Unintentional → diffuse  (affects most blocks globally)

  F4  Delta distribution skewness
        Intentional → heavy right tail  (skewness > 2)
        Unintentional → near-symmetric/mild skew  (skewness < 2)

  F5  Largest connected-component ratio
        Intentional → one dominant cluster covers most tampered blocks
        Unintentional → no dominant cluster

  F6  Edge density of tamper mask
        Intentional → sharp spatial edges at tamper boundary
        Unintentional → no coherent boundary

Each feature votes {INTENTIONAL | UNINTENTIONAL | NEUTRAL}.
Majority vote → final classification + confidence score.

Specific attack sub-types identified
-------------------------------------
  Intentional  → Copy-Paste / Splicing / Inpainting / Object Removal /
                 Content Substitution / Localised Enhancement
  Unintentional → JPEG Compression / Gaussian Noise / Blur /
                  Transmission Error / Global Enhancement

Requirements
------------
    pip install numpy scipy Pillow dtcwt pyswarms matplotlib scikit-image scikit-learn
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.fft import dctn, idctn
from scipy import ndimage, stats
import dtcwt
from dtcwt.numpy import Pyramid
from pyswarms.single.global_best import GlobalBestPSO
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN


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

def _fitness(alpha_mat, HSw_list, wm_sv_list, U_list, Vt_list, dct_blocks):
    lam   = 0.6
    costs = np.empty(len(alpha_mat))
    for i, (alpha,) in enumerate(alpha_mat):
        mse_acc = sv_acc = 0.0
        for hsw, sw, U, Vt, C_orig in zip(
                HSw_list, wm_sv_list, U_list, Vt_list, dct_blocks):
            hsw_new  = hsw + alpha * sw
            C_new    = _isvd(U, hsw_new, Vt)
            mse_acc += np.mean((C_orig - C_new) ** 2)
            sv_acc  += np.mean(np.abs(alpha * sw))
        n        = len(HSw_list)
        avg_mse  = mse_acc / n
        psnr_val = 10.0 * np.log10(255.0 ** 2 / (avg_mse + 1e-12))
        costs[i] = lam * (-psnr_val) + (1.0 - lam) * (sv_acc / n)
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
    I = _load_gray(host_path, M)

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
    Iw_reloaded   = _load_gray(output_path, M)
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
    Iw = _load_gray(watermarked_path, key.M)
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
    img_recv = _load_gray(received_path, key.M)
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

    # ── Algorithm 4: Attack Classification ───────────────────────────────────
    print("[Alg4]  Classifying attack type …")
    classification = classify_attack(
        tamper_grid    = tamper_grid,
        sv_deltas_grid = sv_deltas_grid,
        tampered_frac  = tampered_frac,
        T              = T,
    )

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
                  tampered_frac, is_tampered, n_rows, n_cols,
                  classification)

    return {
        "is_tampered"     : is_tampered,
        "tampered_frac"   : tampered_frac,
        "tamper_grid"     : tamper_grid,
        "sv_deltas"       : sv_deltas_grid,
        "classification"  : classification,
        "tamper_map_path" : tamper_map_path,
        "overlay_path"    : overlay_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 4 – ATTACK CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

# Attack type labels
INTENTIONAL   = "INTENTIONAL"
UNINTENTIONAL = "UNINTENTIONAL"
AUTHENTIC     = "AUTHENTIC"

# Known specific attack sub-types
_INTENTIONAL_SUBTYPES = [
    "Copy-Paste / Splicing",
    "Inpainting / Object Removal",
    "Content Substitution",
    "Localised Intensity Manipulation",
]
_UNINTENTIONAL_SUBTYPES = [
    "JPEG / Lossy Compression",
    "Additive Gaussian Noise",
    "Blur / Low-pass Filtering",
    "Transmission / Channel Error",
    "Global Brightness / Contrast Change",
]


def _feature_clustering(tamper_grid: np.ndarray,
                        sv_deltas_grid: np.ndarray) -> dict:
    """
    F1 – Spatial clustering via DBSCAN on tampered block coordinates.

    Intentional  → tight clusters  (high mean cluster density)
    Unintentional → scattered points  (low density / noise-only)
    """
    rows, cols = np.where(tamper_grid)
    if len(rows) < 2:
        return {"vote": UNINTENTIONAL, "confidence": 0.5,
                "n_clusters": 0, "noise_frac": 1.0,
                "cluster_coords": [], "description": "Too few tampered blocks to cluster"}

    coords = np.column_stack([rows, cols]).astype(float)
    # eps=2 means blocks within 2 grid steps are neighbours
    db     = DBSCAN(eps=2.0, min_samples=2).fit(coords)
    labels = db.labels_
    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac  = float((labels == -1).sum()) / max(len(labels), 1)

    # Collect cluster centroids for visualisation
    cluster_coords = []
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        cluster_coords.append((rows[mask], cols[mask]))

    # Intentional: has at least one cluster AND most tampered blocks
    # belong to clusters (low noise fraction)
    if n_clusters >= 1 and noise_frac < 0.5:
        conf = min(0.95, 0.6 + (1.0 - noise_frac) * 0.35)
        vote = INTENTIONAL
        desc = f"{n_clusters} spatial cluster(s), {noise_frac*100:.0f}% scattered"
    else:
        conf = min(0.90, 0.55 + noise_frac * 0.35)
        vote = UNINTENTIONAL
        desc = f"Diffuse pattern, {noise_frac*100:.0f}% blocks unclustered"

    return {"vote": vote, "confidence": conf,
            "n_clusters": n_clusters, "noise_frac": noise_frac,
            "cluster_coords": cluster_coords, "description": desc}


def _feature_magnitude(sv_deltas: np.ndarray, T: float,
                        tamper_mask: np.ndarray) -> dict:
    """
    F2 – Delta magnitude ratio: mean_tampered_delta / T

    Intentional  → large ratio  (>> 3)  — sharp, high-energy edits
    Unintentional → small ratio  (1–3)  — mild perturbations just above T
    """
    tampered_deltas = sv_deltas[tamper_mask]
    if len(tampered_deltas) == 0:
        return {"vote": UNINTENTIONAL, "confidence": 0.5,
                "magnitude_ratio": 0.0,
                "description": "No tampered blocks"}

    ratio = float(tampered_deltas.mean()) / max(T, 1e-6)
    max_r = float(tampered_deltas.max()) / max(T, 1e-6)

    if ratio > 4.0:
        conf = min(0.95, 0.65 + (ratio - 4.0) * 0.03)
        vote = INTENTIONAL
        desc = f"Mean delta = {ratio:.1f}×T  (strong spike → deliberate edit)"
    elif ratio > 2.0:
        conf = 0.60
        vote = INTENTIONAL
        desc = f"Mean delta = {ratio:.1f}×T  (moderate spike)"
    else:
        conf = min(0.88, 0.55 + (2.0 - ratio) * 0.16)
        vote = UNINTENTIONAL
        desc = f"Mean delta = {ratio:.1f}×T  (mild perturbation)"

    return {"vote": vote, "confidence": conf,
            "magnitude_ratio": ratio, "max_ratio": max_r,
            "description": desc}


def _feature_area_fraction(tampered_frac: float) -> dict:
    """
    F3 – Tampered area fraction.

    Intentional  → localised  (< 30% of blocks)
    Unintentional → diffuse   (> 30% — affects most of the image)

    Note: very large intentional edits (>50%) can break this rule,
    so we use a soft boundary and low confidence.
    """
    if tampered_frac < 0.15:
        vote = INTENTIONAL
        conf = 0.80
        desc = f"{tampered_frac*100:.1f}% of blocks — highly localised"
    elif tampered_frac < 0.30:
        vote = INTENTIONAL
        conf = 0.65
        desc = f"{tampered_frac*100:.1f}% of blocks — moderately localised"
    elif tampered_frac < 0.55:
        vote = UNINTENTIONAL
        conf = 0.65
        desc = f"{tampered_frac*100:.1f}% of blocks — semi-global (likely compression/noise)"
    else:
        vote = UNINTENTIONAL
        conf = 0.85
        desc = f"{tampered_frac*100:.1f}% of blocks — globally distributed"

    return {"vote": vote, "confidence": conf,
            "tampered_frac": tampered_frac, "description": desc}


def _feature_distribution(sv_deltas: np.ndarray,
                           tamper_mask: np.ndarray) -> dict:
    """
    F4 – Statistical distribution of tampered SV deltas.

    Intentional  → heavy right tail  (skewness > 2, high kurtosis)
    Unintentional → near-normal / mild skew
    """
    tampered_deltas = sv_deltas[tamper_mask]
    if len(tampered_deltas) < 4:
        return {"vote": UNINTENTIONAL, "confidence": 0.5,
                "skewness": 0.0, "kurtosis": 0.0,
                "description": "Insufficient samples for distribution analysis"}

    skewness = float(stats.skew(tampered_deltas))
    kurt     = float(stats.kurtosis(tampered_deltas))

    if skewness > 2.0 and kurt > 3.0:
        vote = INTENTIONAL
        conf = min(0.90, 0.65 + skewness * 0.05)
        desc = f"Heavy right tail: skew={skewness:.2f}, kurt={kurt:.2f} → sharp isolated spikes"
    elif skewness > 1.0:
        vote = INTENTIONAL
        conf = 0.62
        desc = f"Mild right skew: skew={skewness:.2f} → some concentrated damage"
    else:
        vote = UNINTENTIONAL
        conf = min(0.85, 0.60 + (2.0 - abs(skewness)) * 0.12)
        desc = f"Near-symmetric: skew={skewness:.2f} → uniform perturbation (noise/compression)"

    return {"vote": vote, "confidence": conf,
            "skewness": skewness, "kurtosis": kurt, "description": desc}


def _feature_largest_component(tamper_grid: np.ndarray) -> dict:
    """
    F5 – Largest connected component (4-connected) as fraction of all
         tampered blocks.

    Intentional  → one big blob  (LCC > 40% of tampered blocks)
    Unintentional → many small isolated blobs
    """
    if tamper_grid.sum() == 0:
        return {"vote": UNINTENTIONAL, "confidence": 0.5,
                "lcc_ratio": 0.0, "description": "No tampered blocks"}

    labeled  = label(tamper_grid, connectivity=1)
    props    = regionprops(labeled)
    areas    = sorted([p.area for p in props], reverse=True)
    lcc_area = areas[0] if areas else 0
    total    = tamper_grid.sum()
    lcc_ratio = lcc_area / max(total, 1)

    if lcc_ratio > 0.5:
        vote = INTENTIONAL
        conf = min(0.92, 0.65 + lcc_ratio * 0.27)
        desc = f"Largest component = {lcc_ratio*100:.0f}% of tampered area → coherent edit region"
    elif lcc_ratio > 0.25:
        vote = INTENTIONAL
        conf = 0.62
        desc = f"Largest component = {lcc_ratio*100:.0f}% → partial clustering"
    else:
        vote = UNINTENTIONAL
        conf = min(0.88, 0.60 + (0.25 - lcc_ratio) * 1.2)
        desc = f"Largest component = {lcc_ratio*100:.0f}% → fragmented (noise-like)"

    return {"vote": vote, "confidence": conf,
            "lcc_ratio": lcc_ratio, "n_components": len(areas),
            "description": desc}


def _feature_edge_density(tamper_grid: np.ndarray) -> dict:
    """
    F6 – Spatial edge density of the tamper mask using Sobel filter.

    Intentional  → sharp edges at tamper boundary  (high edge density)
    Unintentional → no coherent boundary  (low edge density)
    """
    if tamper_grid.sum() == 0:
        return {"vote": UNINTENTIONAL, "confidence": 0.5,
                "edge_density": 0.0, "description": "No tampered blocks"}

    mask_f   = tamper_grid.astype(np.float64)
    gx       = ndimage.sobel(mask_f, axis=1)
    gy       = ndimage.sobel(mask_f, axis=0)
    gradient = np.hypot(gx, gy)
    # Normalise by perimeter of tamper region (# boundary cells)
    perimeter = float((gradient > 0.1).sum())
    area      = float(tamper_grid.sum())
    # Compactness: high = compact shape (intentional), low = scattered
    compactness = perimeter / max(area, 1)

    if compactness < 2.0:
        # Low compactness = compact blob = intentional
        vote = INTENTIONAL
        conf = min(0.90, 0.65 + (2.0 - compactness) * 0.12)
        desc = f"Compact tamper region (compactness={compactness:.2f}) → clean edit boundary"
    elif compactness < 4.0:
        vote = INTENTIONAL
        conf = 0.60
        desc = f"Moderately compact (compactness={compactness:.2f})"
    else:
        vote = UNINTENTIONAL
        conf = min(0.88, 0.60 + (compactness - 4.0) * 0.04)
        desc = f"Scattered/fragmented (compactness={compactness:.2f}) → no clean boundary"

    return {"vote": vote, "confidence": conf,
            "edge_density": perimeter, "compactness": compactness,
            "description": desc}


def _classify_attack_subtype(attack_type: str,
                              features: dict,
                              sv_deltas: np.ndarray,
                              tamper_grid: np.ndarray,
                              T: float) -> str:
    """
    Given the top-level classification, identify the most likely specific
    attack sub-type using feature values.
    """
    if attack_type == AUTHENTIC:
        return "None"

    tamper_mask   = tamper_grid.flatten() > 0
    tampered_d    = sv_deltas.flatten()[tamper_mask]
    mean_delta    = float(tampered_d.mean()) if len(tampered_d) else 0.0
    tampered_frac = features["area"]["tampered_frac"]
    n_clusters    = features["clustering"]["n_clusters"]
    lcc_ratio     = features["lcc"]["lcc_ratio"]
    skewness      = features["distribution"].get("skewness", 0.0)

    if attack_type == INTENTIONAL:
        # Copy-paste / splicing: one large compact cluster, very high delta
        if lcc_ratio > 0.6 and mean_delta > 5 * T:
            return "Copy-Paste / Splicing"
        # Inpainting / object removal: moderate cluster, moderate delta
        if n_clusters <= 2 and 2 * T < mean_delta <= 5 * T:
            return "Inpainting / Object Removal"
        # Content substitution: multiple clusters, high delta
        if n_clusters > 2 and mean_delta > 3 * T:
            return "Content Substitution"
        # Localised intensity: tight cluster, delta just above high threshold
        return "Localised Intensity Manipulation"

    else:  # UNINTENTIONAL
        # JPEG compression: affects most blocks mildly
        if tampered_frac > 0.4 and mean_delta < 3 * T:
            return "JPEG / Lossy Compression"
        # Gaussian noise: diffuse, near-symmetric distribution
        if skewness < 1.0 and tampered_frac > 0.2:
            return "Additive Gaussian Noise"
        # Blur: concentrated in edges/textures (harder to distinguish — use area)
        if tampered_frac < 0.4 and mean_delta < 2.5 * T:
            return "Blur / Low-pass Filtering"
        # Transmission error: random scattered bits
        if n_clusters == 0:
            return "Transmission / Channel Error"
        return "Global Brightness / Contrast Change"


def classify_attack(
    tamper_grid    : np.ndarray,    
    sv_deltas_grid : np.ndarray,
    tampered_frac  : float,
    T              : float,
) -> dict:
    """
    Algorithm 4 – Classify the attack as intentional or unintentional.

    Parameters
    ----------
    tamper_grid    : 2D bool array (n_rows × n_cols)
    sv_deltas_grid : 2D float array of SV distances per block
    tampered_frac  : fraction of blocks flagged as tampered
    T              : tamper threshold

    Returns
    -------
    dict with:
        attack_type  : "INTENTIONAL" | "UNINTENTIONAL" | "AUTHENTIC"
        subtype      : specific attack name
        confidence   : float [0, 1]
        votes        : per-feature votes and confidences
        features     : all extracted feature values
        reasoning    : human-readable explanation
    """

    if tampered_frac == 0.0:
        return {
            "attack_type" : AUTHENTIC,
            "subtype"     : "None",
            "confidence"  : 1.0,
            "votes"       : {},
            "features"    : {},
            "reasoning"   : "No tampered blocks detected. Image is authentic.",
        }

    tamper_mask    = tamper_grid.flatten() > 0
    sv_deltas_flat = sv_deltas_grid.flatten()

    print("[Alg4 / C1]  Extracting classification features …")

    # ── Extract all 6 features ────────────────────────────────────────────────
    f_cluster = _feature_clustering(tamper_grid, sv_deltas_grid)
    f_mag     = _feature_magnitude(sv_deltas_flat, T, tamper_mask)
    f_area    = _feature_area_fraction(tampered_frac)
    f_dist    = _feature_distribution(sv_deltas_flat, tamper_mask)
    f_lcc     = _feature_largest_component(tamper_grid)
    f_edge    = _feature_edge_density(tamper_grid)

    features = {
        "clustering"   : f_cluster,
        "magnitude"    : f_mag,
        "area"         : f_area,
        "distribution" : f_dist,
        "lcc"          : f_lcc,
        "edge"         : f_edge,
    }

    # ── Weighted majority vote ────────────────────────────────────────────────
    # Weights reflect discriminative power of each feature
    feature_weights = {
        "clustering"   : 2.0,   # strongest discriminator
        "lcc"          : 2.0,   # strong
        "magnitude"    : 1.5,
        "area"         : 1.0,
        "distribution" : 1.5,
        "edge"         : 1.0,
    }

    score_intentional   = 0.0
    score_unintentional = 0.0
    votes = {}

    for fname, feat in features.items():
        w    = feature_weights[fname]
        vote = feat["vote"]
        conf = feat["confidence"]
        votes[fname] = {"vote": vote, "confidence": conf,
                        "description": feat["description"]}
        if vote == INTENTIONAL:
            score_intentional   += w * conf
        else:
            score_unintentional += w * conf

    total = score_intentional + score_unintentional
    intentional_prob = score_intentional / max(total, 1e-9)

    if intentional_prob >= 0.55:
        attack_type = INTENTIONAL
        confidence  = float(intentional_prob)
    else:
        attack_type = UNINTENTIONAL
        confidence  = float(1.0 - intentional_prob)

    subtype = _classify_attack_subtype(attack_type, features,
                                        sv_deltas_flat, tamper_grid, T)

    # ── Build reasoning string ────────────────────────────────────────────────
    reasoning_lines = [
        f"Attack type  : {attack_type}  (confidence {confidence*100:.1f}%)",
        f"Sub-type     : {subtype}",
        "",
        "Feature votes:",
    ]
    for fname, v in votes.items():
        arrow = "→ INT" if v["vote"] == INTENTIONAL else "→ UNI"
        reasoning_lines.append(
            f"  [{fname:<14}]  {arrow}  conf={v['confidence']:.2f}  |  {v['description']}"
        )

    reasoning = "\n".join(reasoning_lines)
    print("[Alg4 / C2]  Classification complete.")
    print(reasoning)

    return {
        "attack_type" : attack_type,
        "subtype"     : subtype,
        "confidence"  : confidence,
        "votes"       : votes,
        "features"    : features,
        "reasoning"   : reasoning,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _plot_summary(recv_img, tmap_img, sv_deltas_grid, T,
                  tampered_frac, is_tampered, n_rows, n_cols,
                  classification: dict):
    """
    5-panel figure:
      1. Received image
      2. Tamper map (green/red)
      3. SV-distance heatmap with threshold contour
      4. Feature vote radar / bar chart
      5. Classification verdict panel
    """
    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor("#0f172a")
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                             height_ratios=[1, 0.35],
                             hspace=0.35, wspace=0.35)

    verdict_color = "#ef4444" if is_tampered else "#22c55e"
    verdict_text  = "⚠  IMAGE TAMPERED" if is_tampered else "✓  IMAGE AUTHENTIC"

    atk   = classification.get("attack_type", "N/A")
    sub   = classification.get("subtype", "N/A")
    conf  = classification.get("confidence", 0.0)
    atk_color = "#f97316" if atk == INTENTIONAL else \
                "#60a5fa"  if atk == UNINTENTIONAL else "#22c55e"

    # ── Panel 1: Received image ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(recv_img, cmap="gray")
    ax1.set_title("Received Image", color="white", fontsize=11, pad=8)
    ax1.axis("off")

    # ── Panel 2: Tamper map ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(tmap_img)
    ax2.set_title("Tamper Map", color="white", fontsize=11, pad=8)
    ax2.axis("off")
    ax2.legend(
        handles=[mpatches.Patch(color="#22c55e", label="Authentic"),
                 mpatches.Patch(color="#ef4444", label="Tampered")],
        loc="lower right", fontsize=8,
        facecolor="#1e293b", labelcolor="white", framealpha=0.8,
    )

    # ── Panel 3: SV-distance heatmap ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    im  = ax3.imshow(sv_deltas_grid, cmap="hot", interpolation="nearest",
                     aspect="auto")
    ax3.set_title(f"SV-Distance Heatmap  (T={T:.2f})",
                  color="white", fontsize=11, pad=8)
    ax3.set_xlabel(f"{tampered_frac*100:.1f}% blocks tampered",
                   color="#94a3b8", fontsize=9)
    ax3.tick_params(colors="white")
    cb = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    try:
        ax3.contour(sv_deltas_grid, levels=[T], colors=["cyan"], linewidths=1.2)
    except Exception:
        pass

    # ── Panel 4: Feature confidence bar chart ────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    votes = classification.get("votes", {})
    if votes:
        fnames  = list(votes.keys())
        confs   = [votes[f]["confidence"] for f in fnames]
        colors  = ["#f97316" if votes[f]["vote"] == INTENTIONAL
                   else "#60a5fa" for f in fnames]
        bars    = ax4.barh(fnames, confs, color=colors, height=0.55)
        ax4.set_xlim(0, 1)
        ax4.axvline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
        ax4.set_title("Feature Votes", color="white", fontsize=11, pad=8)
        ax4.tick_params(colors="white", labelsize=8)
        ax4.set_xlabel("Confidence", color="#94a3b8", fontsize=9)
        for spine in ax4.spines.values():
            spine.set_edgecolor("#334155")
        ax4.set_facecolor("#1e293b")
        # Annotate bars
        for bar, fname in zip(bars, fnames):
            label_txt = "INT" if votes[fname]["vote"] == INTENTIONAL else "UNI"
            ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     label_txt, va="center", color="white", fontsize=7)
    else:
        ax4.text(0.5, 0.5, "No features\n(authentic image)",
                 ha="center", va="center", color="white", fontsize=10)
        ax4.axis("off")

    # ── Panel 5: Classification verdict ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.set_facecolor("#1e293b")
    ax5.axis("off")
    ax5.set_title("Classification", color="white", fontsize=11, pad=8)

    ax5.text(0.5, 0.88, atk, ha="center", va="center",
             transform=ax5.transAxes, fontsize=18, fontweight="bold",
             color=atk_color,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a",
                       edgecolor=atk_color, linewidth=2))
    ax5.text(0.5, 0.65, sub, ha="center", va="center",
             transform=ax5.transAxes, fontsize=10, color="#e2e8f0",
             style="italic")
    ax5.text(0.5, 0.48, f"Confidence: {conf*100:.1f}%",
             ha="center", va="center",
             transform=ax5.transAxes, fontsize=11, color=atk_color)

    # Confidence gauge bar
    gauge_x = np.linspace(0.1, 0.9, 100)
    ax5.fill_between(gauge_x,
                     [0.32] * 100, [0.38] * 100,
                     transform=ax5.transAxes,
                     color="#334155")
    ax5.fill_between(gauge_x[:int(conf * 100)],
                     [0.32] * int(conf * 100),
                     [0.38] * int(conf * 100),
                     transform=ax5.transAxes,
                     color=atk_color)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#334155")

    # ── Bottom banner ─────────────────────────────────────────────────────────
    ax_bot = fig.add_subplot(gs[1, :])
    ax_bot.set_facecolor("#1e293b")
    ax_bot.axis("off")
    ax_bot.text(0.5, 0.7, verdict_text, ha="center", va="center",
                transform=ax_bot.transAxes, fontsize=15,
                fontweight="bold", color=verdict_color)
    ax_bot.text(0.5, 0.25,
                f"Attack: {atk}  |  Sub-type: {sub}  |  "
                f"Confidence: {conf*100:.1f}%  |  "
                f"Tampered area: {tampered_frac*100:.1f}%",
                ha="center", va="center",
                transform=ax_bot.transAxes, fontsize=10, color="#94a3b8")

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
        clf     = result["classification"]
        print(f"\n{'='*60}")
        print(f"  Verdict    : {verdict}  ({result['tampered_frac']*100:.1f}% blocks flagged)")
        print(f"  Attack     : {clf['attack_type']}")
        print(f"  Sub-type   : {clf['subtype']}")
        print(f"  Confidence : {clf['confidence']*100:.1f}%")
        print(f"{'='*60}")

    else:
        parser.print_help()
        sys.exit(1)