
import traceback
from .models import ImageProcess
from .embedding import embed_watermark


def run_pipeline(process_id):
    process = ImageProcess.objects.get(id=process_id)

    try:
        # 🔹 STEP 1: Resize
        process.set_status(ImageProcess.Status.RESIZING, 10)

        # 🔹 STEP 2+: Call your pipeline
        # IMPORTANT: Modify embed_watermark to accept process_id
        embed_watermark(
            host_path=process.original_image.path,
            watermark_path=process.watermark_image.path,
            process_id=process.id   
        )

        # 🔹 FINAL
        process.set_status(ImageProcess.Status.COMPLETED, 100)

    except Exception as e:
        traceback.print_exc()
        process.mark_failed(str(e))