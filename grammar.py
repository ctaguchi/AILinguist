from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
import glob
import os
from pydantic import BaseModel, Field
from pypdf import PdfReader
import fitz

load_dotenv()

type Message = Dict[str, str | Dict | List[Dict[str, str]]]

class Grammar:
    def __init__(self,
                 model: str = "gpt-4o-2024-08-06",
                 system_prompt: str = "You are a professional summarizer of a grammar textbook. You will be given a text extracted from a PDF file of a textbook about the grammar of the Jinghpaw language.",
                 user_prompt: str = "Please summarize the textbook provided below. Please provide words, example sentences, and their English translations as much as possible so that users can learn Jinghpaw from your summary."):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

        self.client = OpenAI()
        self.messages: List[Message] = [{"role": "system",
                                         "content": system_prompt},
                                        {"role": "user",
                                         "content": user_prompt}]

    def summarize(self, message):
        """Summarize (no structured output)."""
        messages = self.messages + message
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
            )
        return response.choices[0].message.content

    def read_pdf(self, file_path: str) -> str:
        text = ""
        reader = PdfReader(file_path)
        for p in reader.pages:
            text += p.extract_text()
        return text

    def read_and_summarize_pdf(self, file_path: str) -> list:
        """Summarize while reading to reduce the context length."""
        texts = []
        summaries = []
        # reader = PdfReader(file_path)
        doc = fitz.open(file_path)
        for i, p in enumerate(doc):
        # for i, p in enumerate(reader.pages):
            # page = p.extract_text()
            page = p.get_text()
            # print(page)
            texts.append(page)
            if (i + 1) % 10 == 0:
                text = "\n".join(texts)
                message = [{"role": "user",
                            "content": text}]
                summary = self.summarize(message)
                print(summary)
                summaries.append(summary)
                texts = []
        return summaries

    
    def extra_summarize(self,
                        file_path: str = "grammar_summary.txt",
                        outfile: str = "grammar_summary_extra.txt") -> None:
        """Further summarize a summarized output.
        `file_path` should point to a .txt file that stores the summarized output."""
        with open(file_path, "r") as f:
            summary = f.read()

        system_prompt = "You are a professional linguist. You will be given a grammar description of the Jinghpaw language."
        user_prompt = "Please summarize the grammar description provided below, focusing on morphology and syntax."
        user_prompt += f"\n\n{summary}"

        messages = [{"role": "system",
                     "content": system_prompt},
                    {"role": "user",
                     "content": user_prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
            )
        content = response.choices[0].message.content
        with open(outfile, "w") as f:
            f.write(content)
            

if __name__ == "__main__":
    system_prompt = "You are a professional summarizer of a grammar textbook. You will be given a text extracted from a PDF file of a textbook about the grammar of the Jinghpaw language."
    user_prompt = "Please summarize the textbook provided below. Please provide words, example sentences, and their English translations as much as possible so that users can learn Jinghpaw from your summary."
    model = "gpt-4o-2024-08-06"
    grammar = Grammar(model=model,
                      system_prompt=system_prompt,
                      user_prompt=user_prompt)

    grammar_file = "grammar.pdf"
    # text = grammar.read_pdf(grammar_file)
    # grammar.messages[1]["content"] += ("\n\n" + text)
    summaries = grammar.read_and_summarize_pdf(grammar_file)

    # summary = grammar.summarize()
    # print(summary)

    with open("grammar_summary.txt", "w") as f:
        f.write("\n".join(summaries))
