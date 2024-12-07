from pdf2image import convert_from_path
from typing import Literal
import os

def convert(pdf_path: str,
            out_dir: str,
            mode: Literal["png", "jpg"] = "png") -> None:
    """Convert pdf to images."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print("Converting the images...")
    images = convert_from_path(pdf_path)
    for i in range(len(images)):
        print("Saving page", i)
        if mode == "jpg":
            filename = "page" + str(i) + ".jpg"
            filepath = os.path.join(out_dir, filename)
            images[i].save(filepath)
        elif mode == "png":
            filename = "page" + str(i) + ".png"
            filepath = os.path.join(out_dir, filename)
            images[i].save(filepath)
    print("Save complete")

if __name__ == "__main__":
    convert("dictionary.pdf",
            "jpeg_images",
            mode="jpg")
