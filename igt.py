from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
import glob
import os
from pydantic import BaseModel, Field
import copy
import argparse

load_dotenv()

GRAMMAR_SUMMARY = "./grammar_summary.txt"

type Message = Dict[str, str | Dict | List[Dict[str, str]]]

class Gloss(BaseModel):
    words: list[str]
    gloss: list[str]

class IGT:
    def __init__(self,
                 model: str,
                 system_prompt: str = "You are a professional linguist. You will be given a Jinghpaw sentence, and your task is to provide the interlinear glossed text (IGT) for the sentence.",
                 user_prompt: str = "Please analyze the following Jinghpaw sentence and provide its IGT in the specified format: {}",
                 grammar_prompt: bool = False,
                 dictionary: bool = False):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.dictionary = dictionary
        self.model = model

        self.client = OpenAI()
        self.messages: List[Message] = [{"role": "system",
                                         "content": system_prompt},
                                        {"role": "user",
                                         "content": user_prompt}]

        if grammar_prompt:
            self.user_prompt += f"\n\n{grammar_prompt}"
            

    def ask(self, sent: str):
        """Summarize (no structured output)."""
        if self.dictionary:
            # TODO: retrieve from the dictionary
            ...
            
        messages = self.messages.format(sent)
        
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=Gloss
            )
        response = completion.choices[0].message #.parsed
        return response

    def igt_for_file(self, file_path: str):
        """Perform IGT for a text file containing one sentence per line."""
        with open(file_path, "r") as f:
            sents = f.readlines()

        for sent in sents:
            response = self.ask(sent)
            content: dict = json.loads(response.content)


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_file",
                        type=str,
                        help="Path to the input text file, which contains one sentence per line.")
    parser.add_argument("-d",
                        "--dictionary",
                        action="store_true",
                        help="If true, relevant words from the dictionary will be included in the prompt.")
    parser.add_argument("-g",
                        "--grammar",
                        action="store_true",
                        help="If true, a grammar summary will be included in the prompt.")
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = get_args()
    model = "gpt-4o-2024-08-06"

    system_prompt = "You are a professional linguist. You will be given a Jinghpaw sentence, and your task is to provide the interlinear glossed text (IGT) for the sentence."
    user_prompt = "Please analyze the following Jinghpaw sentence and provide its IGT in the specified format: {}"

    if args.dictionary:
        # TODO: retrieve relevant dictionary entries from `retriever.py`.
        ...
        
    if args.grammar:
        # Include a grammar summary.
        with open(GRAMMAR_SUMMARY, "r") as f:
            grammar = f.read()
        grammar_prompt = f"When analyzing the sentence, you may also take into account the grammar summary provided below:\n{grammar}"
    else:
        grammar_prompt = None
        
    igt = IGT(model, system_prompt, user_prompt, grammar_prompt, args.dictionary)
    igt.igt_for_file(args.input_file)
    
    # with open("dictionary.json", "w") as f:
        # json.dump(entries, f)
