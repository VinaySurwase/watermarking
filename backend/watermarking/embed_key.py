"""
embed_key.py  (UPDATED)
───────────────────────
Dataclass that wraps the .npz key produced at embedding time and consumed
at extraction / tamper-detection time.

Change vs original:
  - Added `Sw_full` field (FIX BUG 4): the full singular value vector of the
    encrypted watermark W_enc. Without this, extraction had to zero-pad from
    sv_len=8 up to wm_side, badly distorting every ISVD reconstruction.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbedKey:
    # ── Core embedding parameters ──────────────────────────────────────────
    alpha_star:       float
    block_size:       int
    dtcwt_levels:     int
    henon_a:          float
    henon_b:          float
    M:                int

    # ── Watermark SVD components ────────────────────────────────────────────
    Uw:               np.ndarray          # left singular vectors  of W_enc
    Sw_full:          np.ndarray          # FIX BUG 4: full SV vector of W_enc
    Vtw:              np.ndarray          # right singular vectors of W_enc
    watermark_shape:  tuple

    # ── Host image data ─────────────────────────────────────────────────────
    HSw_list:         np.ndarray          # (n_blocks, sv_len) float64  FIX BUG 2
    HSw_new_dominant: np.ndarray          # dominant SV per block after embedding

    # ── Tamper detection ────────────────────────────────────────────────────
    tamper_threshold: float

    # ── Padding metadata ────────────────────────────────────────────────────
    orig_H:           int
    orig_W:           int
    pad_h:            int
    pad_w:            int

    # Optional — only present when host was padded
    bottom_pad:       Optional[np.ndarray] = field(default=None)
    right_pad:        Optional[np.ndarray] = field(default=None)
    corner_pad:       Optional[np.ndarray] = field(default=None)

    # ── Convenience ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, npz_path: str) -> "EmbedKey":
        """
        Load an EmbedKey from a .npz file written by embed_watermark().
        Handles both new keys (with Sw_full) and legacy keys (without).
        """
        data = np.load(npz_path, allow_pickle=False)

        def _opt_arr(name: str) -> Optional[np.ndarray]:
            """Return array or None if key missing or array is empty."""
            if name not in data:
                return None
            arr = data[name]
            return arr if arr.size > 0 else None

        # FIX BUG 4: load Sw_full; fall back gracefully for legacy keys that
        # pre-date this fix (Sw_full will be reconstructed as zeros — extraction
        # will still be poor, but won't crash).
        if "Sw_full" in data:
            Sw_full = data["Sw_full"].astype(np.float64)
        else:
            # Legacy key: best-effort — use zeros of inferred size
            Uw_tmp = data["Uw"]
            Sw_full = np.zeros(min(Uw_tmp.shape), dtype=np.float64)

        # FIX BUG 2: force float64 regardless of how the array was saved
        HSw_list = data["HSw_list"].astype(np.float64)

        return cls(
            alpha_star       = float(data["alpha_star"]),
            block_size       = int(data["block_size"]),
            dtcwt_levels     = int(data["dtcwt_levels"]),
            henon_a          = float(data["henon_a"]),
            henon_b          = float(data["henon_b"]),
            M                = int(data["M"]),
            Uw               = data["Uw"].astype(np.float64),
            Sw_full          = Sw_full,
            Vtw              = data["Vtw"].astype(np.float64),
            watermark_shape  = tuple(data["watermark_shape"].tolist()),
            HSw_list         = HSw_list,
            HSw_new_dominant = data["HSw_new_dominant"].astype(np.float64),
            tamper_threshold = float(data["tamper_threshold"]),
            orig_H           = int(data["orig_H"]),
            orig_W           = int(data["orig_W"]),
            pad_h            = int(data["pad_h"]),
            pad_w            = int(data["pad_w"]),
            bottom_pad       = _opt_arr("bottom_pad"),
            right_pad        = _opt_arr("right_pad"),
            corner_pad       = _opt_arr("corner_pad"),
        )