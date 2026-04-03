from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
import threading
from django.shortcuts import get_object_or_404
from .utils_runner import run_pipeline
from .models import ImageProcess


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upload_images(request):
    try:
        original = request.FILES.get("original_image")
        watermark = request.FILES.get("watermark_image")

        # 🔴 Validation
        if not original or not watermark:
            return Response(
                {"error": "Both images are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 🔴 File type validation
        allowed_types = ["image/png", "image/jpeg", "image/jpg"]

        if original.content_type not in allowed_types:
            return Response(
                {"error": "Invalid original image format"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if watermark.content_type not in allowed_types:
            return Response(
                {"error": "Invalid watermark image format"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # ✅ Step 1: Create process (IMPORTANT for ID/paths)
        process = ImageProcess.objects.create(
            user=request.user,
            status=ImageProcess.Status.PENDING
        )

        # ✅ Step 2: Assign files
        process.original_image = original
        process.watermark_image = watermark
        process.save()

        return Response({
            "message": "Upload successful",
            "process_id": process.id,
            "status": process.status,
            "original_image": process.original_image.url,
            "watermark_image": process.watermark_image.url
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_process(request, process_id):
    process = get_object_or_404(
        ImageProcess,
        id=process_id,
        user=request.user
    )

    if process.status not in [ImageProcess.Status.PENDING, ImageProcess.Status.FAILED]:
        return Response({
            "error": "Process already started"
        }, status=status.HTTP_400_BAD_REQUEST)

    process.set_status(ImageProcess.Status.RESIZING, 5)

    thread = threading.Thread(
        target=run_pipeline,
        args=(process.id,),
        daemon=True   # 🔥 important
    )
    thread.start()

    return Response({
        "message": "Processing started",
        "process_id": process.id,
        "status": process.status
    })