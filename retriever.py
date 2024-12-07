from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, TypeAlias
import glob
import os
from pydantic import BaseModel, Field
import copy
import json
import random

load_dotenv()

DICTIONARY = "dictionary.json"
Hint: TypeAlias = Dict[str, Dict[str, List[str]] | List[Dict[str, str]]]

type Message = Dict[str, str | Dict | List[Dict[str, str]]]

class Entry(BaseModel):
    example: str
    english_translation: str

class Entries(BaseModel):
    examples: list[Entry]

class Retriever:
    def __init__(self,
                 model: str = "gpt-4o-2024-08-06",
                 system_prompt: str = "Your task is to retrieve relevant headwords and example sentences from the given dictionary. You will be given a sentence and a dictionary in the JSON format.",
                 user_prompt: str = "From the dictionary, please retrieve Jinghpaw headwords and  their example sentences that appear in the input sentence, as well as their English translation. For each headword, please provide a few example sentences. Input sentence: "):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

        self.client = OpenAI()
        self.messages: List[Message] = [{"role": "system",
                                         "content": system_prompt},
                                        {"role": "user",
                                         "content": user_prompt}]
        with open(DICTIONARY, "r") as f:
            self.dictionary = json.load(f)

            
    def retrieve_llm(self, sent: str) -> str:
        """Summarize (no structured output)."""
        messages = copy.deepcopy(self.messages)
        messages[1]["content"] += (sent + "\n\nDictionary:\n" + str(self.dictionary))

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=Entries
            )
        response = completion.choices[0].message #.parsed
        content = response.content
        return str(content)


    def retrieve(self,
                 sent: str,
                 num_samples: int = 5) -> List[Hint]:
        """Retrieve dictionary examples."""
        words = sent.lower().split()
        hints = []
        for word in words:
            definitions = []
            examples = []
            for entry in self.dictionary["entries"]:
                headword = entry["headword"].replace("q", "")
                modified = entry["headword"].replace("g", "k")
                if word == headword:
                    definitions += entry["english_definitions"] # List
                    examples += entry["examples"] # List[Dict[str, str]]
                elif word == modified:
                    definitions += entry["english_definitions"]
                    examples += entry["examples"]

                # pick up num_samples examples
                if word == headword or word == modified:
                    if len(examples) > num_samples:
                        examples = random.sample(examples, num_samples)
                        
            if not definitions and not examples:
                continue
            hints.append({"word": {"headword": headword,
                                   "definitions": list(set(definitions))},
                          "examples": examples})
        return hints
