from django.urls import path
from .upload_view import *
from .extract_view import *
from .verify_view import *
from .compare_view import compare_images

urlpatterns = [
    path("upload/", upload_images, name="upload_images"),
    
    path("process/<int:process_id>/status",get_process_status),
    
    path("process/<int:process_id>/resizing/", get_resizing_step),
    path("process/<int:process_id>/forward/", get_forwarding_step),
    path("process/<int:process_id>/encryption/", get_encryption_step),
    path("process/<int:process_id>/svd/", get_svd_step),
    path("process/<int:process_id>/pso/", get_pso_step),
    path("process/<int:process_id>/embedding/", get_embedding_step),
    path("process/<int:process_id>/threshold/", get_threshold_step),

    # images
    path("process/<int:process_id>/image/original/", get_original_image),
    path("process/<int:process_id>/image/resized/", get_resized_image),
    path("process/<int:process_id>/image/wm_raw/", get_watermark_raw),
    path("process/<int:process_id>/image/wm_encrypted/", get_watermark_encrypted),
    path("process/<int:process_id>/image/output/", get_output_image),

    # key
    path("process/<int:process_id>/key/", download_key),
]

urlpatterns += [
    path("verify/submit/", submit_verify, name="submit_verify"),

    path(
        "verify/<int:verification_id>/status/",
        get_verify_status,
        name="get_verify_status",
    ),

    path(
        "verify/<int:verification_id>/result/",
        get_verify_result,
        name="get_verify_result",
    ),

    path(
        "verify/<int:verification_id>/image/received/",
        get_verify_received_image,
        name="get_verify_received_image",
    ),
    path(
        "verify/<int:verification_id>/image/tamper_map/",
        get_verify_tamper_map,
        name="get_verify_tamper_map",
    ),
    path(
        "verify/<int:verification_id>/image/overlay/",
        get_verify_overlay,
        name="get_verify_overlay",
    ),
]

urlpatterns += [
    path("extract/submit/", submit_extract, name="submit_extract"),
 
    path(
        "extract/<int:extraction_id>/status/",
        get_extract_status,
        name="get_extract_status",
    ),
 
    path(
        "extract/<int:extraction_id>/result/",
        get_extract_result,
        name="get_extract_result",
    ),
 
    path(
        "extract/<int:extraction_id>/image/input/",
        get_extract_input_image,
        name="get_extract_input_image",
    ),
 
    path(
        "extract/<int:extraction_id>/image/output/",
        get_extract_output_image,
        name="get_extract_output_image",
    ),
]

# ── Compare ──────────────────────────────────────────────────────
urlpatterns += [
    path("compare/", compare_images, name="compare_images"),
]