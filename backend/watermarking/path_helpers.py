import os
import uuid

def get_filename(filename):
    """
    Generate unique filename while preserving extension
    """
    ext = os.path.splitext(filename)[1]  # includes dot (.png)

    if not ext:
        ext = ".png"  # fallback (important for your pipeline)

    return f"{uuid.uuid4().hex}{ext}"

def path_original(instance):
    return f"storage/images/user_{instance.user.id}/process_{instance.id}/original_image.png"


def path_watermark(instance):
    return f"storage/watermark/user_{instance.user.id}/process_{instance.id}/watermark_image.png"


def path_resized(instance):
    return f"storage/resized/user_{instance.user.id}/process_{instance.id}/resized_image.png"


def path_wm_raw(instance):
    return f"storage/watermark_raw/user_{instance.user.id}/process_{instance.id}/watermark_raw.png"


def path_wm_encrypted(instance):
    return f"storage/watermark_encrypted/user_{instance.user.id}/process_{instance.id}/watermark_encrypted.png"


def path_output(instance):
    return f"storage/output/user_{instance.user.id}/process_{instance.id}/watermarked_image.png"

def path_original_watermarked(instance):
    return f"storage/original_watermarked/user_{instance.user.id}/process_{instance.id}/original_watermarked_image.png"

def path_key(instance):
    return f"storage/keys/user_{instance.user.id}/process_{instance.id}/key"


def path_extract_input(instance):
    """Uploaded watermarked image submitted for extraction."""
    return (
        f"storage/extract/user_{instance.user.id}"
        f"/extract_{instance.id}/watermarked_input.png"
    )


def path_extract_output(instance):
    """Recovered watermark image written by extract_watermark()."""
    return (
        f"storage/extract/user_{instance.user.id}"
        f"/extract_{instance.id}/extracted_watermark.png"
    )


def path_verify_received(instance):
    """Uploaded (possibly tampered) image to verify."""
    return (
        f"storage/verify/user_{instance.user.id}"
        f"/verify_{instance.id}/received_image.png"
    )


def path_verify_tamper_map(instance):
    """Green/red block map produced by verify_tamper."""
    return (
        f"storage/verify/user_{instance.user.id}"
        f"/verify_{instance.id}/tamper_map.png"
    )


def path_verify_overlay(instance):
    """Received image with red overlay on tampered regions."""
    return (
        f"storage/verify/user_{instance.user.id}"
        f"/verify_{instance.id}/tamper_overlay.png"
    )

def path_reconstructed_watermark(instance):
    """Received image with red overlay on tampered regions."""
    return (
        f"storage/reconstructed_watermark/user_{instance.user.id}"
        f"/reconstructed_watermark{instance.id}/reconstructed_watermark.png"
    )