"""
verify_runner.py
────────────────
Background task that loads the .npz key, rebuilds an EmbedKey,
calls verify_tamper(), and writes all results back to the
TamperVerification row.

Usage (called from the view via threading.Thread):
    from .verify_runner import run_verify_pipeline
    thread = threading.Thread(
        target=run_verify_pipeline,
        args=(verification.id,),
        daemon=True,
    )
    thread.start()
"""

import traceback
import numpy as np

from .models import (
    TamperVerification,
    path_verify_received,
    path_verify_tamper_map,
    path_verify_overlay,
    path_key,
)
from .embedding import EmbedKey
from .verify_tamper import verify_tamper


def load_key(npz_path: str) -> EmbedKey:
    """Reconstruct EmbedKey from the .npz file saved during embedding."""
    data = np.load(npz_path, allow_pickle=True)

    return EmbedKey(
        alpha_star        = float(data["alpha_star"]),
        HSw_new_dominant  = data["HSw_new_dominant"],
        tamper_threshold  = float(data["tamper_threshold"]),
        Uw                = data["Uw"],
        Vtw               = data["Vtw"],
        watermark_shape   = tuple(data["watermark_shape"].tolist()),
        henon_a           = float(data["henon_a"]),
        henon_b           = float(data["henon_b"]),
        M                 = int(data["M"]),
        block_size        = int(data["block_size"]),
        dtcwt_levels      = int(data["dtcwt_levels"]),
        HSw_list          = tuple(data["HSw_list"].tolist()),
        orig_H            = int(data["orig_H"]),
        orig_W            = int(data["orig_W"]),
        pad_h             = int(data["pad_h"]),
        pad_w             = int(data["pad_w"]),
        bottom_pad        = data["bottom_pad"],
        right_pad         = data["right_pad"],
    )


def run_verify_pipeline(verification_id: int) -> None:
    """
    Entry point for the background thread.

    Steps
    -----
    1. Load the TamperVerification row.
    2. Load the .npz key (from the linked ImageProcess or from a
       user-supplied key path stored on the instance).
    3. Call verify_tamper().
    4. Persist all results to the DB row.
    """
    verification = TamperVerification.objects.get(id=verification_id)

    try:
        # ── 1. Mark as running ───────────────────────────────
        verification.set_status(TamperVerification.Status.VERIFYING, 10)

        # ── 2. Resolve key path ──────────────────────────────
        if verification.source_process is None:
            raise ValueError(
                "No source_process linked – cannot locate the .npz key."
            )

        key_path = path_key(verification.source_process) + ".npz"
        key      = load_key(key_path)

        verification.set_status(TamperVerification.Status.VERIFYING, 30)

        # ── 3. Resolve received-image path ───────────────────
        received_path = path_verify_received(verification)
        tamper_map    = path_verify_tamper_map(verification)
        overlay       = path_verify_overlay(verification)

        # ── 4. Run algorithm ─────────────────────────────────
        result = verify_tamper(
            received_path   = received_path,
            key             = key,
            tamper_map_path = tamper_map,
            overlay_path    = overlay,
        )

        verification.set_status(TamperVerification.Status.VERIFYING, 80)

        # ── 5. Persist results ───────────────────────────────
        grid   = result["tamper_grid"]      # 2-D bool array
        deltas = result["sv_deltas"]        # 2-D float array

        verification.is_tampered          = result["is_tampered"]
        verification.tampered_frac        = result["tampered_frac"]
        verification.tamper_grid_flat     = grid.astype(int).flatten().tolist()
        verification.grid_rows            = int(grid.shape[0])
        verification.grid_cols            = int(grid.shape[1])
        verification.sv_deltas_flat       = deltas.flatten().tolist()
        verification.tamper_threshold_used = float(key.tamper_threshold)
        verification.set_status(TamperVerification.Status.COMPLETED, 100)

    except Exception as exc:
        traceback.print_exc()
        verification.mark_failed(str(exc))