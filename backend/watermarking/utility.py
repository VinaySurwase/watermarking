import numpy as np
from django.core.files.base import ContentFile
import io

def save_key_npz(key_data: dict):
    """
    key_data = {
        'alpha_star': float,
        'HSw_list': [...],
        ...
    }
    """

    buffer = io.BytesIO()

    np.savez(buffer, **key_data)

    buffer.seek(0)

    return ContentFile(buffer.read(), name="key.npz")