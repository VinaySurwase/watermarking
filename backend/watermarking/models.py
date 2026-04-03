from django.db import models
from django.contrib.auth.models import User
import os
import uuid


def _get_filename(filename):
    ext = filename.split('.')[-1]
    return f"{uuid.uuid4().hex}.{ext}"


def upload_original(instance, filename):
    return f"images/user_{instance.user.id}/process_{instance.id}/{_get_filename(filename)}"


def upload_watermark(instance, filename):
    return f"watermark/user_{instance.user.id}/process_{instance.id}/{_get_filename(filename)}"


def upload_resized(instance, filename):
    return f"resized/user_{instance.user.id}/process_{instance.id}/{_get_filename(filename)}"


def upload_wm_raw(instance, filename):
    return f"watermark_raw/user_{instance.user.id}/process_{instance.id}/{_get_filename(filename)}"


def upload_wm_encrypted(instance, filename):
    return f"watermark_encrypted/user_{instance.user.id}/process_{instance.id}/{_get_filename(filename)}"


def upload_output(instance, filename):
    return f"output/user_{instance.user.id}/process_{instance.id}/{_get_filename(filename)}"


def upload_key(instance, filename):
    return f"keys/user_{instance.user.id}/process_{instance.id}/{_get_filename(filename)}"



class ImageProcess(models.Model):

    # =========================================================
    # 🔹 STATUS ENUM (CLEAN PIPELINE CONTROL)
    # =========================================================
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"

        RESIZING = "resizing", "Resizing Image"
        FORWARDING = "forwarding", "Running Forward Pipeline"
        ENCRYPTING = "encrypting", "Encrypting Watermark"
        SVD = "svd", "Applying SVD"
        PSO = "pso", "Optimizing with PSO"
        EMBEDDING = "embedding", "Embedding Watermark"

        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    status = models.CharField(
        max_length=50,
        choices=Status.choices,
        default=Status.PENDING
    )

    # =========================================================
    # 🔹 USER
    # =========================================================
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    # =========================================================
    # 🔹 INPUT IMAGES
    # =========================================================
    original_image = models.ImageField(upload_to=upload_original)
    watermark_image = models.ImageField(upload_to = upload_watermark)

    # =========================================================
    # 🔹 INTERMEDIATE IMAGES
    # =========================================================
    resized_image = models.ImageField(upload_to=upload_resized, null=True, blank=True)

    # Watermark stages
    watermark_raw = models.ImageField(upload_to=upload_wm_raw, null=True, blank=True)   
    watermark_encrypted = models.ImageField(upload_to=upload_wm_encrypted, null=True, blank=True)  

    # =========================================================
    # 🔹 OUTPUT
    # =========================================================
    watermarked_image = models.ImageField(upload_to=upload_output, null=True, blank=True)

    # =========================================================
    # 🔹 ALGORITHM PARAMETERS
    # =========================================================
    alpha_star = models.FloatField(null=True, blank=True)
    tamper_threshold = models.FloatField(null=True, blank=True)

    # PSO
    pso_particles = models.IntegerField(null=True, blank=True)
    pso_iterations = models.IntegerField(null=True, blank=True)
    pso_cost = models.FloatField(null=True, blank=True)

    # =========================================================
    # 🔹 PIPELINE METADATA
    # =========================================================
    dtcwt_levels = models.IntegerField(null=True, blank=True)

    n_blocks = models.IntegerField(null=True, blank=True)
    sv_length = models.IntegerField(null=True, blank=True)

    LL_shape_h = models.IntegerField(null=True, blank=True)
    LL_shape_w = models.IntegerField(null=True, blank=True)
    
    max_benign_drift = models.FloatField(null = True,blank = True)
    auto_threshold = models.FloatField(null = True,blank = True)
    final_threshold = models.FloatField(null = True,blank = True)
    watermark_shape = models.JSONField(null=True, blank=True)
    
    

    # =========================================================
    # 🔹 WATERMARK INFO
    # =========================================================

    henon_a = models.FloatField(null=True, blank=True)
    henon_b = models.FloatField(null=True, blank=True)

    # =========================================================
    # 🔹 KEY STORAGE
    # =========================================================
    key_file = models.FileField(upload_to=upload_key, null=True, blank=True)


    # =========================================================
    # 🔹 METRICS
    # =========================================================
    psnr_value = models.FloatField(null=True, blank=True)

    # =========================================================
    # 🔹 TIMESTAMP
    # =========================================================
    created_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(null=True, blank=True)
    progress = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Process {self.id} - {self.user.username}"
    
    
