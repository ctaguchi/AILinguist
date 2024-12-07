from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
import glob
import os
from pydantic import BaseModel, Field
import fitz
import argparse
from collections import deque
import json
import copy

load_dotenv()

type Message = Dict[str, str | Dict | List[Dict[str, str]]]

class Example(BaseModel):
    example_sentence: str
    english_translation: str

class Entry(BaseModel):
    headword: str
    english_definitions: list[str]
    examples: list[Example]

class Dictionary(BaseModel):
    headwords: list[Entry]
    # headword: str | None = None
    # content: Entry | None = None

# debug:
# print(json.dumps(Example.model_json_schema(), indent=4))
# print(json.dumps(Entry.model_json_schema(), indent=4))
# print(json.dumps(Dictionary.model_json_schema(), indent=4))

class Summarizer:
    def __init__(self,
                 model: str,
                 system_prompt: str,
                 user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

        self.client = OpenAI(
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("PROJECT_ID"),
        )
        self.messages: List[Message] = [{"role": "system",
                                         "content": system_prompt},
                                        {"role": "user",
                                         "content": user_prompt}]

    def summarize(self):
        """Summarize (no structured output)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            )
        return response.choices[0].message.content

    def summarize_structured(self, message):
        """Summarize with the structured output."""
        messages = copy.deepcopy(self.messages)
        print(messages)
        messages[1]["content"] += ("\n\n" + message)
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=Dictionary
            )
        response = completion.choices[0].message #.parsed
        print(response)
        return response
        
    def read_book(self, folder: str) -> str:
        """Read book pages (txt files) in a folder."""
        folder_files = os.path.join(folder, "*")
        pages = glob.glob(folder_files)
        text = ""
        for p in pages:
            with open(p, "r") as f:
                text += f.read()
        return text

    def read_pdf(self, pdf: str) -> str:
        """Read book (pdf)."""
        text = ""
        doc = fitz.open(file_path)
        for i, p in enumerate(doc):
            page = p.get_text() + "\n"
            text += page
        return text

    def read_summarize_pdf(self, pdf: str) -> str:
        """Summarize the dictionary (pdf) page by page.
        Process it like a queue so that the LLM has access to the
        previous and succeeding page."""
        queue = deque()
        doc = fitz.open(pdf)
        entries = dict()
        for i, p in enumerate(doc):
            page = p.get_text()                
            queue.append(page)
            if len(queue) == 1:
                # the first page; skip
                continue
            elif len(queue) == 2:
                # the second page; queue[0]: current, queue[1]: next
                prev_page = "None"
                curr_page = queue[0]
                next_page = queue[1]
            elif len(queue) == 3:
                # other pages
                prev_page = queue.popleft()
                curr_page = queue[0]
                next_page = queue[1]
            else:
                raise ValueError(f"The length of the queue has to be in the range of 1 to 3 but it's {len(queue)}.")
            prev_prompt = f"Previous page:\n{prev_page}\n\n"
            curr_prompt = f"Current page:\n{curr_page}\n\n"
            next_prompt = f"Next page:\n{next_page}"
            message = prev_prompt + curr_prompt + next_prompt
            entry = self.summarize_structured(message)
            print(entry)
            if entry['headword'] is None:
                continue
            entries.update(entry)
        return entries

    def read_summarize_tex(self, tex: str) -> dict:
        """Summarize the dictionary (tex)."""
        # entries = dict()
        entries = list()
        queue = deque()
        with open(tex, "r") as f:
            lines = f.readlines()

        def divide_chunks(l, n):
            # looping till length l
            for i in range(0, len(l), n): 
                yield l[i:i + n]
                
        # chunking by 100 lines
        chunks = list(divide_chunks(lines, 100))
        
        for i, chunk in enumerate(chunks):
            # i: int, chunk: list
            queue.append("".join(chunk))
            if len(queue) == 1:
                continue
            elif len(queue) == 2:
                prev_page = "None"
            elif len(queue) == 3:
                prev_page = queue.popleft()
            else:
                raise ValueError(f"The length of the queue has to be in the range of 1 to 3 but it's {len(queue)}.")
            curr_page = queue[0]
            next_page = queue[1]

            message = f"Previous 100 lines:\n{prev_page}\n\n" \
                f"Current 100 lines:\n{curr_page}\n\n" \
                f"Next 100 lines:\n{next_page}"
            # print(message)
            response = self.summarize_structured(message)
            content: dict = json.loads(response.content)
            print(content)
            # print(f"{type(response)=}")
            # print(f"{response.content=}")
            # print(response.parsed)
            # print(f"{type(response.parsed)=}")
            if response.parsed.headwords is None:
                continue
            for headword in content["headwords"]:
                # entries.update(headword)
                entries.append(headword)
                # 3print(entries)
                
            with open("checkpoint.json", "w") as f:
                entries_dict = {"entries": entries}
                json.dump(entries_dict, f)
            # entries.update(content)
        entries_dict = {"entries": entries}
        return entries_dict


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr",
                        action="store_true",
                        help="If true, the summarizer will read a OCR-generated text.")
    parser.add_argument("--tex",
                        action="store_true",
                        help="If true, the summarizer will read a tex file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model = "gpt-4o-2024-08-06"

    if args.ocr:
        system_prompt = "You are a professional summarizer of dictionaries. You will be given a text which is an OCR result of an entire Jinghpaw-Chinese dictionary."
        user_prompt = "Please summarize the OCR output below."
        summarizer = Summarizer(model, system_prompt, user_prompt)
        folder = "./text"
        text = summarizer.read_book(folder)
        summarizer.summarize_structured(text)
    elif args.tex:
        system_prompt = "You are a professional summarizer of dictionaries. You will be given a fragment from a tex file of a Jinghpaw-Japanese dictionary."
        user_prompt = r"Please summarize the dictionary entries in the specified format. Please ignore the preface texts. If the orthography of the entry and the word in its examples do not match, follow the orthography of the example. For your information, the previous and the following 100 lines are also provided."
        summarizer = Summarizer(model, system_prompt, user_prompt)
        tex = "dictxionary.txt"
        entries = summarizer.read_summarize_tex(tex)

        with open("dictionary.json", "w") as f:
            json.dump(entries, f)
    else:
        system_prompt = "You are a professional summarizer of dictionaries. You will be given a text of a Jinghpaw-Chinese dictionary."
        user_prompt = r"Please summarize the dictionary page shown below in the specified format. Please ignore the preface texts; return `None` in this case. If the orthography of the entry and the word in its examples do not match, follow the orthography of the example. For your information, the previous and following pages are also provided."
        summarizer = Summarizer(model, system_prompt, user_prompt)
        pdf = "dictionary.pdf"
        entries_dict = summarizer.read_summarize_pdf(pdf)

        with open("dictionary.json", "w") as f:
            json.dump(entries_dict, f)
