import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_dicom(dicom_path):
    """Load DICOM file"""
    if not os.path.exists(dicom_path):
        raise FileNotFoundError("DICOM file not found!")

    ds = pydicom.dcmread(dicom_path)
    return ds


def print_metadata(ds):
    """Print important metadata"""
    print("\n========== METADATA ==========")

    print("Patient Name      :", getattr(ds, "PatientName", "N/A"))
    print("Patient ID        :", getattr(ds, "PatientID", "N/A"))
    print("Modality          :", getattr(ds, "Modality", "N/A"))
    print("Study Date        :", getattr(ds, "StudyDate", "N/A"))
    print("Image Dimensions  :", getattr(ds, "Rows", "N/A"), "x", getattr(ds, "Columns", "N/A"))

    print("==============================\n")


def extract_image(ds):
    """Extract pixel array"""
    try:
        image = ds.pixel_array
    except Exception as e:
        raise RuntimeError("Cannot extract pixel data. Install pylibjpeg/gdcm") from e

    return image


def normalize_image(image):
    """Normalize image to 0-255"""
    image = image.astype(np.float32)

    image -= np.min(image)
    image /= (np.max(image) + 1e-8)

    image = (image * 255).astype(np.uint8)
    return image


def show_image(image, title="DICOM Image"):
    """Display image"""
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_png(image, output_path="output.png"):
    """Save image as PNG"""
    cv2.imwrite(output_path, image)
    print(f"✅ Image saved to {output_path}")


def dump_all_tags(ds):
    """Print all DICOM tags"""
    print("\n========== ALL TAGS ==========")
    for elem in ds:
        print(elem)
    print("==============================\n")


def main():
    dicom_path = "sample1.dcm"   

    # 1. Load
    ds = load_dicom(dicom_path)

    # 2. Metadata
    print_metadata(ds)

    # 3. Extract image
    image = extract_image(ds)

    # 4. Normalize
    norm_image = normalize_image(image)

    # 5. Show
    show_image(norm_image)

    # 6. Save PNG
    save_png(norm_image, "output.png")

    # 7. Dump all tags (optional)
    dump_all_tags(ds)


main()