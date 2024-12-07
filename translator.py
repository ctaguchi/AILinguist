from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
import glob
import os
from pydantic import BaseModel, Field
import copy
import json
from retriever import Retriever

load_dotenv()

DICTIONARY = "dictionary.json"
# GRAMMAR = "grammar_summary.txt" # too long a context length
GRAMMAR = "grammar_summary_extra.txt"

type Message = Dict[str, str | Dict | List[Dict[str, str]]]

class Gloss(BaseModel):
    jinghpaw: str = Field(description="Jinghpaw word or morpheme.")
    english: str = Field(description="English translation of the Jinghpaw word or morpheme.")

class TranslationGloss(BaseModel):
    igt: list[Gloss] = Field(description="List of Jinghpaw-English translation pairs for each word in the sentence.")
    translation: str = Field(description="English translation of the sentence.")

class Translation(BaseModel):
    translation: str = Field(description="English translation of the sentence.")

class Translator:
    def __init__(self,
                 model: str = "gpt-4o-2024-08-06",
                 system_prompt: str = "You are a professional Jinghpaw-English translator. You will be given a Jinghpaw sentence, and your task is to translate it into English.",
                 user_prompt: str = None,
                 igt: bool = False,
                 dictionary: bool = False,
                 grammar: bool = False,
                 structured: bool = False,
                 llm_retrieval: bool = False
                 ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

        self.client = OpenAI()
        self.messages: List[Message] = [{"role": "system",
                                         "content": system_prompt}]

        self.igt = igt
        self.has_dictionary = dictionary
        if self.has_dictionary:
            DICTIONARY = "dictionary.json"
            with open(DICTIONARY, "r") as f:
                self.dictionary = json.load(f)
        self.grammar = grammar
        self.structured = structured
        self.llm_retrieval = llm_retrieval
        
        
    def translate(self, sent: str) -> dict | str:
        """Translate the provided sentence."""
        user_prompt = f"Please translate the following Jinghpaw sentence into English: {sent}"

        if self.igt:
            # incorporated; CoT
            user_prompt += "\nPlease provide an interlinear-glossed text (IGT) of the sentence first, and then give the translation."

        if self.has_dictionary:
            retriever = Retriever()
            if self.llm_retrieval:
                examples = retriever.retrieve_llm(sent)
                user_prompt += f"\n\nFor your information, I provide example sentences that are relevant to the given sentence:\n{examples}"
            else:
                examples = retriever.retrieve(sent)
                user_prompt += f"\n\nFor your information, I provide examples that are relevant to the given sentence:\n{examples}"
            
        if self.grammar:
            with open(GRAMMAR, "r") as f:
                grammar_summary = f.read()
            user_prompt += f"\n\nAlso, you may refer to the concise grammar description of Jinghpaw provided below to analyze the sentence: \n{grammar_summary}"

        user_message = {"role": "user",
                        "content": user_prompt}

        # print(f"{user_message}")
        if self.structured:
            response_format = TranslationGloss if self.igt else Translation
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=self.messages + [user_message],
                response_format=response_format
                )
            return json.loads(response.choices[0].message.content)
        else:            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages + [user_message]
            )
            return response.choices[0].message.content
