import json
import numpy as np
from dataclasses import dataclass
from PIL import Image
from scipy.fft import dctn, idctn
import dtcwt
from dtcwt.numpy import Pyramid
from pyswarms.single.global_best import GlobalBestPSO
import cv2
import math
import argparse

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

def apply_dtcwt(img):
    
    # Initialize DTCWT transform (2D)
    transform = dtcwt.Transform2d()

    # Apply 3-level transform
    coeffs = transform.forward(img, nlevels=3)

    # Extract low-frequency subband at level 3 (LL3)
    LL3 = coeffs.lowpass

    return LL3

def divide_into_blocks(image):
    H, W = image.shape

    # Ensure dimensions are multiples of 8
    assert H % 8 == 0 and W % 8 == 0, "Image size must be multiple of 8"

    blocks = []

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = image[i:i+8, j:j+8]
            blocks.append(block)

    return blocks

def apply_dct_to_blocks(blocks):
    dct_blocks = []

    for block in blocks:
        # Convert to float32 (IMPORTANT)
        block = block.astype(np.float32)

        # Apply 2D DCT
        dct_block = cv2.dct(block)

        dct_blocks.append(dct_block)

    return dct_blocks


def svd_and_cache(dct_blocks):
    H, W, _, _ = dct_blocks.shape

    # Cache structures
    U_cache = np.zeros_like(dct_blocks)
    Vt_cache = np.zeros_like(dct_blocks)
    HSw = np.zeros((H, W, 8))  # singular values

    for i in range(H):
        for j in range(W):
            C = dct_blocks[i, j]

            # SVD
            U, S, Vt = np.linalg.svd(C, full_matrices=False)

            # Store results
            U_cache[i, j] = U
            Vt_cache[i, j] = Vt
            HSw[i, j] = S   # DiagVec(Σ)

    return U_cache, Vt_cache, HSw

def henon_encrypt(W, a=1.4, b=0.3, x0=0.1, y0=0.1):
    H, W_ = W.shape
    size = H * W_

    x, y = x0, y0
    seq = []

    for _ in range(size):
        x_new = 1 - a * x * x + y
        y_new = b * x
        x, y = x_new, y_new
        seq.append(x)

    seq = np.array(seq)

    # Normalize to [0,1]
    seq = (seq - seq.min()) / (seq.max() - seq.min())

    # Convert to permutation
    indices = np.argsort(seq)

    flat = W.flatten()
    encrypted = flat[indices]

    return encrypted.reshape(H, W_), indices

def watermark_svd(W_enc):
    U_w, S_w, Vt_w = np.linalg.svd(W_enc, full_matrices=False)
    return U_w, S_w, Vt_w

def compute_psnr(I, Iw):
    mse = np.mean((I - Iw) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10((255**2) / mse)

def compute_nc(W, W_extracted):
    num = np.sum(W * W_extracted)
    den = np.sqrt(np.sum(W**2) * np.sum(W_extracted**2))
    return num / den if den != 0 else 0

def optimize_alpha(I,LL3, blocks, U_cache, Vt_cache, HSw, Sw, coeffs):
    import numpy as np

    w1, w2 = 0.7, 0.3  # prioritize PSNR slightly more

    particles = np.random.uniform(0.01, 0.1, 6)
    velocities = np.zeros(6)

    pbest = particles.copy()
    gbest = particles[0]

    # def fitness(alpha):
    #     # Embed watermark
    #     modified_blocks = embed_watermark(U_cache, Vt_cache, HSw, Sw, alpha)

    #     H_b, W_b, _, _ = blocks.shape
    #     LL3_new = merge_blocks(modified_blocks, H_b, W_b)

    #     Iw = inverse_dtcwt(LL3_new, coeffs)
    #     Iw = np.clip(Iw, 0, 255)

    #     # PSNR
    #     mse = np.mean((I - Iw) ** 2)
    #     psnr = 100 if mse == 0 else 10 * np.log10((255**2) / mse)

    #     # Approx NC using singular values stability
    #     nc = np.mean(np.abs(Sw[:3])) / (np.mean(np.abs(Sw[:3])) + 1e-6)

    #     return w1 * psnr + w2 * nc
    def fitness(alpha):
        modified_blocks = embed_watermark(U_cache, Vt_cache, HSw, Sw, alpha)

        H_b, W_b, _, _ = blocks.shape
        LL3_new = merge_blocks(modified_blocks, H_b, W_b)

        # FAST PSNR (no inverse DTCWT)
        Hc, Wc = LL3_new.shape
        mse = np.mean((LL3_new - LL3[:Hc, :Wc]) ** 2)
        psnr = 100 if mse == 0 else 10 * np.log10((255**2) / mse)

        nc = np.mean(np.abs(Sw[:3]))

        return w1 * psnr + w2 * nc

    # Initialize gbest
    for i in range(len(particles)):
        if fitness(particles[i]) > fitness(gbest):
            gbest = particles[i]

    # PSO loop
    for _ in range(8):
        for i in range(len(particles)):
            f = fitness(particles[i])

            if f > fitness(pbest[i]):
                pbest[i] = particles[i]

            if f > fitness(gbest):
                gbest = particles[i]

            velocities[i] = (
                0.5 * velocities[i]
                + 0.8 * (pbest[i] - particles[i])
                + 0.9 * (gbest - particles[i])
            )

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0.01, 0.1)

    return gbest


def embed_watermark(U_cache, Vt_cache, HSw, Sw, alpha):
    H, W, _ = HSw.shape

    modified_blocks = []

    for i in range(H):
        for j in range(W):
            # Step 16
            S_new = HSw[i, j] + alpha * Sw[:8]

            # Step 17
            S_mat = np.diag(S_new)

            # Step 18
            C_prime = U_cache[i, j] @ S_mat @ Vt_cache[i, j]

            # Step 19 (IDCT)
            block = cv2.idct(C_prime) + 128

            modified_blocks.append(block)

    return modified_blocks

def merge_blocks(blocks, H_blocks, W_blocks):
    img = np.zeros((H_blocks*8, W_blocks*8))

    idx = 0
    for i in range(H_blocks):
        for j in range(W_blocks):
            img[i*8:(i+1)*8, j*8:(j+1)*8] = blocks[idx]
            idx += 1

    return img

def inverse_dtcwt(LL3_new, coeffs):
    transform = dtcwt.Transform2d()

    reconstructed = transform.inverse(
        dtcwt.Pyramid(
            lowpass=LL3_new,
            highpasses=coeffs.highpasses
        )
    )

    return reconstructed
    

def blocks_4d(image):
    H, W = image.shape
    return image.reshape(H//8, 8, W//8, 8).swapaxes(1, 2)

# def blocks_4d(image):
#     H, W = image.shape

#     # Crop to multiple of 8
#     H_new = (H // 8) * 8
#     W_new = (W // 8) * 8

#     image = image[:H_new, :W_new]

#     return image.reshape(H_new//8, 8, W_new//8, 8).swapaxes(1, 2)

def apply_dct_4d(blocks):
    H, W, _, _ = blocks.shape
    dct_blocks = np.zeros_like(blocks)

    for i in range(H):
        for j in range(W):
            block = blocks[i, j].astype(np.float32) - 128
            dct_blocks[i, j] = cv2.dct(block)

    return dct_blocks


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--watermark", required=True)
    parser.add_argument("--output", default="watermarked.png")
    parser.add_argument("--enc_wm", default="encrypted_wm.png")
    parser.add_argument("--key", default="key.json")

    args = parser.parse_args()

    # ========= STEP 1 =========
    print("Step 1")
    I = resize(args.image)

    # ========= STEP 2 =========
    print("Step 2")
    
    transform = dtcwt.Transform2d()
    coeffs = transform.forward(I, nlevels=3)
    LL3 = coeffs.lowpass

    # ========= STEP 3 =========
    print("Step 3")
    
    blocks = blocks_4d(LL3)

    # ========= STEP 4 =========
    print("Step 4")

    dct_blocks = apply_dct_4d(blocks)

    # ========= STEP 5 =========
    print("Step 5")
    
    U_cache, Vt_cache, HSw = svd_and_cache(dct_blocks)

    # ========= STEP 6 =========
    print("Step 6")
    
    W = resize(args.watermark)
    W = cv2.resize(W, (8, 8))
    W = W.astype(np.float32) / 255.0

    # ========= STEP 7 =========
    print("Step 7")
    
    a, b = 1.4, 0.3
    x0, y0 = 0.1, 0.1
    W_enc, indices = henon_encrypt(W, a, b, x0, y0)

    # Save encrypted watermark
    W_enc_img = (W_enc * 255).astype(np.uint8)
    cv2.imwrite(args.enc_wm, W_enc_img)

    # ========= STEP 8 =========
    print("Step 8")
    
    U_w, Sw, Vt_w = watermark_svd(W_enc)

    # ========= STEP 9: OPTIMIZED ALPHA =========
    print("Step 9")
    
    alpha = optimize_alpha(
        I,LL3, blocks,
        U_cache, Vt_cache, HSw, Sw,
        coeffs
    )

    print("✅ Optimized alpha:", alpha)

    # ========= STEP 10 =========
    print("Step 10")
    
    modified_blocks = embed_watermark(U_cache, Vt_cache, HSw, Sw, alpha)

    # ========= STEP 11 =========
    print("Step 11")
    
    H_b, W_b, _, _ = blocks.shape
    LL3_new = merge_blocks(modified_blocks, H_b, W_b)

    # ========= STEP 12 =========
    print("Step 12")
    
    Iw = inverse_dtcwt(LL3_new, coeffs)
    Iw = np.clip(Iw, 0, 255).astype(np.uint8)

    # ========= SAVE =========
    print("Save")
    
    cv2.imwrite(args.output, Iw)

    key_data = {
        "a": a,
        "b": b,
        "x0": x0,
        "y0": y0,
        "alpha": float(alpha),
        "watermark_shape": list(W.shape),
        "indices": indices.tolist()
    }

    with open(args.key, "w") as f:
        json.dump(key_data, f)

    print("✅ Watermarked image saved:", args.output)
    print("✅ Encrypted watermark saved:", args.enc_wm)
    print("✅ Key saved:", args.key)
    
main()
