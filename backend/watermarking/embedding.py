import numpy as np
from dataclasses import dataclass,field
from typing import List, Tuple
from PIL import Image
from scipy.fft import dctn, idctn
import dtcwt
from dtcwt.numpy import Pyramid
from pyswarms.single.global_best import GlobalBestPSO
import io

# ══════════════════════════════════════════════════════════════════════════════
#  KEY BUNDLE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbedKey:
    alpha_star        : float
    HSw_new_dominant  : np.ndarray        # shape (n_blocks,)  float64
    tamper_threshold  : float
    Uw                : np.ndarray
    Vtw               : np.ndarray
    watermark_shape   : Tuple[int, int]
    wm_sv_list        : List[np.ndarray] = field(default_factory=list)
    HSw_list          : List[np.ndarray] = field(default_factory=list)
    henon_a           : float = 1.4
    henon_b           : float = 0.3
    M                 : int   = 512
    block_size        : int   = 8
    dtcwt_levels      : int   = 3
    
    orig_H            : int = 0
    orig_W            : int = 0
    pad_h             : int = 0
    pad_w             : int = 0

    bottom_pad        : np.ndarray = None
    right_pad         : np.ndarray = None


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

import cv2
import math
def resize(image_path):

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

def resize2(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found or invalid path")

    H, W = img.shape

    M = math.ceil(max(H, W) / 64) * 64

    pad_h = M - H
    pad_w = M - W

    padded = np.pad(
        img,
        ((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0
    )

    return padded.astype(np.float32), (H, W, pad_h, pad_w)

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

def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

import cv2
from django.core.files.base import ContentFile

def numpy_to_png_file(img: 'np.ndarray'):
    """
    Convert numpy array to Django ContentFile (PNG format)
    
    Args:
        img: numpy array (grayscale or RGB)
        filename: name of output file
    
    Returns:
        (filename, ContentFile)
    """
    success, buffer = cv2.imencode('.png', img)
    
    if not success:
        raise ValueError("Failed to encode image")

    return  ContentFile(buffer.tobytes(),"image")

from .models import *
import os

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    success = cv2.imwrite(path, img)
    if not success:
        raise ValueError("Failed to save image")
def embed_watermark(
    host_path: str,
    watermark_path: str,
    process_id: int,
    M: int = 512,
    block_size: int = 8,
    dtcwt_levels: int = 3,
    henon_a: float = 1.4,
    henon_b: float = 0.3,
    pso_particles: int = 20,
    pso_iters: int = 50,
    alpha_bounds: tuple = (0.001, 0.05),
    tamper_threshold: float = 3.0,
    output_path: str = "watermarked.png",
) -> tuple:

    from .models import ImageProcess
    import numpy as np

    process = ImageProcess.objects.get(id=process_id)

    try:
        # =====================================================
        # 🔹 STEP 1: RESIZE
        # =====================================================
        process.set_status(ImageProcess.Status.RESIZING, 10)
        process.save()

        I, pad_info = resize2(host_path)
        orig_H, orig_W, pad_h, pad_w = pad_info

        # save resized image
        # filename = get_filename("resized.png")
        # process.resized_image = numpy_to_png_file(I),
        save_image(path_resized(process),I)
            
        process.dtcwt_levels = dtcwt_levels
        process.save()

        # =====================================================
        # 🔹 STEP 2: FORWARD PIPELINE
        # =====================================================
        process.set_status(ImageProcess.Status.FORWARDING, 25)

        U_list, HSw_list, Vt_list, dct_blocks, positions, LL, highpasses, tr = \
            _forward_pipeline(I, block_size, dtcwt_levels)

        n_blocks = len(HSw_list)
        LL_shape = LL.shape
        sv_len = len(HSw_list[0])

        process.n_blocks = n_blocks
        process.sv_length = sv_len
        process.LL_shape_h = LL_shape[0]
        process.LL_shape_w = LL_shape[1]
        process.save()

        # =====================================================
        # 🔹 STEP 3: ENCRYPTION
        # =====================================================
        process.set_status(ImageProcess.Status.ENCRYPTING, 40)

        wm_side = max(int(np.sqrt(n_blocks)) * block_size, block_size)
        W_raw = _load_gray(watermark_path, wm_side)
        W_enc = henon_encrypt(W_raw, a=henon_a, b=henon_b)

        # save W_raw
        # filename = get_filename("watermark_raw.png")
        # process.watermark_raw.save(filename, numpy_to_png_file(W_raw),True)

        # # save W_enc
        # filename = get_filename("watermark_encrypted.png")
        # process.watermark_encrypted.save(filename, numpy_to_png_file(W_enc),True)
        save_image(path_wm_raw(process),W_raw)
        save_image(path_wm_encrypted(process),W_enc)

        process.henon_a = henon_a
        process.henon_b = henon_b
        process.watermark_shape = list(W_enc.shape)
        process.save()

        # =====================================================
        # 🔹 STEP 4: SVD
        # =====================================================
        process.set_status(ImageProcess.Status.SVD, 55)
       

        Uw, Sw_full, Vtw = _svd(W_enc)

        total_needed = n_blocks * sv_len
        Sw_tiled = np.tile(Sw_full, int(np.ceil(total_needed / len(Sw_full))))

        wm_sv_list = [
            Sw_tiled[i * sv_len: i * sv_len + sv_len].copy()
            for i in range(n_blocks)
        ]

        # =====================================================
        # 🔹 STEP 5: PSO
        # =====================================================
        process.set_status(ImageProcess.Status.PSO, 70)

        optimizer = GlobalBestPSO(
            n_particles=pso_particles,
            dimensions=1,
            options={"c1": 0.5, "c2": 0.3, "w": 0.9},
            bounds=(np.array([alpha_bounds[0]]), np.array([alpha_bounds[1]])),
        )

        cost, best = optimizer.optimize(
            lambda a: _fitness(a, HSw_list, wm_sv_list, U_list, Vt_list, dct_blocks),
            iters=pso_iters, verbose=False,
        )

        alpha_star = float(best[0])

        process.alpha_star = alpha_star
        process.pso_cost = float(cost)
        process.pso_particles = pso_particles
        process.pso_iterations = pso_iters
        process.save()

        # =====================================================
        # 🔹 STEP 6: EMBEDDING
        # =====================================================
        process.set_status(ImageProcess.Status.EMBEDDING, 85)
        

        new_dct_blocks = []
        HSw_new_dominant = np.empty(n_blocks, dtype=np.float64)

        for idx, (hsw, sw, U, Vt) in enumerate(zip(HSw_list, wm_sv_list, U_list, Vt_list)):
            hsw_new = hsw + alpha_star * sw
            C_new = _isvd(U, hsw_new, Vt)
            new_dct_blocks.append(C_new)
            HSw_new_dominant[idx] = hsw_new[0]

        Iw = _inverse_pipeline(new_dct_blocks, positions, LL_shape, highpasses, tr, block_size)
        Iw_uint8 = np.clip(Iw, 0, 255).astype(np.uint8)
        H, W = Iw_uint8.shape

        # Extract padded regions
        bottom_pad = Iw_uint8[orig_H:H, :] if pad_h > 0 else None
        right_pad  = Iw_uint8[:, orig_W:W] if pad_w > 0 else None
        corner_pad = Iw_uint8[orig_H:H, orig_W:W] if (pad_h > 0 and pad_w > 0) else None
        
        Iw_cropped = Iw_uint8[:orig_H, :orig_W]

        # save output image
        # filename = get_filename("output.png")
        # process.watermarked_image.save(filename, numpy_to_png_file(Iw_uint8),True)
        # save_image(path_output(process),Iw_uint8)
        save_image(path_output(process), Iw_cropped)

        _psnr = psnr(I, Iw_uint8.astype(np.float64))
        process.psnr_value = float(_psnr)
        process.save()

        # =====================================================
        # 🔹 STEP 7: THRESHOLD
        # =====================================================
        process.set_status(ImageProcess.Status.THRESHOLDING, 95)
        
        Iw_reloaded = resize(path_output(process))

        _, sv_reload, _, _, _, _, _, _ = _forward_pipeline(
            Iw_reloaded, block_size, dtcwt_levels
        )

        drift = np.array([
            abs(sv_reload[i][0] - HSw_new_dominant[i])
            for i in range(n_blocks)
        ])

        max_benign_drift = float(drift.max())
        auto_threshold = max_benign_drift * 2.5
        final_threshold = max(tamper_threshold, auto_threshold)

        process.max_benign_drift = max_benign_drift
        process.auto_threshold = auto_threshold
        process.final_threshold = final_threshold
        process.tamper_threshold = final_threshold

        # =====================================================
        # 🔹 SAVE KEY (.npz)
        # =====================================================
        key_path = path_key(process)

        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        
        np.savez(
            key_path,
            alpha_star=alpha_star,
            HSw_new_dominant=HSw_new_dominant,
            tamper_threshold=final_threshold,
            Uw=Uw,
            Vtw=Vtw,
            watermark_shape=W_enc.shape,
            henon_a=henon_a,
            henon_b=henon_b,
            M=M,
            block_size=block_size,
            dtcwt_levels=dtcwt_levels,
            HSw_list=np.array(HSw_list, dtype=object), 
            
             # padding metadata
            orig_H=orig_H,
            orig_W=orig_W,
            pad_h=pad_h,
            pad_w=pad_w,

            # ✅ ACTUAL CUT DATA
            bottom_pad=bottom_pad,
            right_pad=right_pad,
            corner_pad=corner_pad
        )
        

        # =====================================================
        # 🔹 COMPLETE
        # =====================================================
        process.set_status(ImageProcess.Status.COMPLETED, 100)

        return Iw_uint8, None

    except Exception as e:
        process.mark_failed(str(e))
        raise

