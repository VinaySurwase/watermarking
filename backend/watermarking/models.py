from django.db import models
from django.contrib.auth.models import User

class ImageProcess(models.Model):
    # 🔹 User
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    # 🔹 Input Images
    original_image = models.ImageField(upload_to='images/')
    watermark_image = models.ImageField(upload_to='watermark/')

    # 🔹 Intermediate Images
    resized_image = models.ImageField(upload_to='resized/', null=True, blank=True)

    # 🔹 Output Image
    watermarked_image = models.ImageField(upload_to='output/', null=True, blank=True)

    # 🔹 Status
    status = models.CharField(max_length=50, default="pending")

    # 🔹 Parameters
    alpha = models.FloatField(null=True, blank=True)
    threshold = models.FloatField(null=True, blank=True)

    # 🔹 Key (.npz file)
    key_file = models.FileField(upload_to='keys/', null=True, blank=True)

    # 🔹 Key Metadata (optional but useful)
    alpha_star = models.FloatField(null=True, blank=True)
    tamper_threshold = models.FloatField(null=True, blank=True)

    # 🔹 Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Process {self.id} - {self.user.username}"