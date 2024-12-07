# AI Linguist

## Setup
Install the dependencies by running `poetry install` in the directory.
If you do not have `poetry` installed, follow the instructions in the documentation: https://python-poetry.org/docs/

1. Summarize the grammar: Run `grammar.py` to summarize the grammar book.
1. Summarize the dictionary: Run `summarizer.py` to extract the dictionary entries in the structured output.
1. Retrieve the dictionary examples: Run `retriever.py` to extract the relevant dictionary entries and example sentences. Make sure that you have the dictionary summarized already.
1. Translate: Run `translator.py` to translate a sentence of your target language.