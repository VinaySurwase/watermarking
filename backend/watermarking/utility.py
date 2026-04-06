import numpy as np
from django.core.files.base import ContentFile
import io
import cv2

def reconstruct_full_image(cropped, key):
    # return cropped
    orig_H = key.orig_H
    orig_W = key.orig_W
    pad_h  = key.pad_h
    pad_w  = key.pad_w

    H = orig_H + pad_h
    W = orig_W + pad_w

    full = np.zeros((H, W), dtype=cropped.dtype)

    # place original
    full[:orig_H, :orig_W] = cropped

    # restore pads
    if pad_h > 0:
        full[orig_H:H, :] = key.bottom_pad

    if pad_w > 0:
        full[:, orig_W:W] = key.right_pad

    return full


def getImg(image_path):

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found or invalid path")

    return img