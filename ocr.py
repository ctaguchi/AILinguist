from openai import OpenAI
from dotenv import load_dotenv
import base64
from typing import List, Dict, Optional
import glob
import os
from pathlib import Path
import argparse

load_dotenv()

type Message = Dict[str, str | Dict | List[Dict[str, str]]]

class OCR:
    def __init__(self,
                 model: str,
                 system_prompt: str,
                 user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

        self.client = OpenAI()
        self.messages: List[Message] = [{"role": "system",
                                         "content": {"type": "text",
                                                     "text": system_prompt}},
                                        {"role": "user",
                                         "content": [{"type": "text",
                                                      "text": user_prompt}]}]
        
        
    def encode_image(self, image_path: str):
        """Encode an image into base64 format for a request."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def api(self, image_path: str):
        """Throw a request to LLM."""
        base64_image = self.encode_image(image_path)
        image_message = {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        user_message = self.messages[1]
        user_message["content"].append(image_message)
        system_message = self.messages[0]
        messages = [system_message, user_message]
        # print(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
            )
        return response.choices[0].message.content

    def save_output(self, text: str, out_path: str) -> None:
        """Save the output text."""
        with open(out_path, "w") as f:
            f.write(text)

    def ocr_from_folder(self,
                        image_folder: str,
                        out_folder: str,
                        same_stem: bool = True,
                        start_from: Optional[int] = None):
        """Iterate over all the images in the specified image folder."""
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            
        # targets = glob.glob(os.path.join(image_folder, "*"))
        num_files = len(glob.glob(os.path.join(image_folder, "*")))
        # for t in targets:
        pages = range(num_files) if start_from is None else range(start_from, num_files)
        for i in pages:
            target = os.path.join(image_folder, f"page{i}.jpg")
            print(f"Processing image:", target)
            output = self.api(target)
            if same_stem:
                out_path = os.path.join(out_folder, Path(target).stem + ".txt")
                self.save_output(output, out_path)
            else:
                raise NotImplementedError


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_from", type=int,
                        help="Specify the page number to start from. Optional.")
    return parser.parse_args()

            
def debug():
    """For debugging with one page."""
    system_prompt = "The provided image is a page from a Jinghpaw-Chinese dictionary."
    user_prompt = "Please do OCR. Please only include the text in the image."
    model = "gpt-4o-2024-08-06"
    ocr = OCR(model, system_prompt, user_prompt)
    text = ocr.api("jpeg_images/page194.jpg")
    ocr.save_output(text, "text/page194.txt")

    
if __name__ == "__main__":
    args = get_args()
    
    system_prompt = "The provided image is a page from a Jinghpaw-Chinese dictionary."
    user_prompt = "Please do OCR. Please only include the text in the image."
    model = "gpt-4o-2024-08-06"
    ocr = OCR(model, system_prompt, user_prompt)

    image_folder = "jpeg_images"
    out_folder = "text"
    ocr.ocr_from_folder(image_folder, out_folder, start_from=args.start_from)
