from typing import Dict, Literal
from dotenv import load_dotenv
from openai import OpenAI
import argparse
import pandas as pd
import os

# local imports
from translator import Translator
from igt import IGT
from auto_eval import evaluate

load_dotenv()

GRAMMAR_SUMMARY = "./grammar_summary.txt"

DEV_KAC = "./data/dev.kac_Latn"
DEV_ENG = "./data/dev.eng_Latn"

CONV_KAC = "./data/kac_conv.kac"
CONV_ENG = "./data/kac_conv.en"

def load_dev(data: Literal["flores","conv"]) -> Dict[str, list]:
    if data == "flores":
        with open(DEV_KAC, "r") as f:
            dev_kac = f.readlines()
        with open(DEV_ENG, "r") as f:
            dev_eng = f.readlines()
    elif data == "conv":
        with open(CONV_KAC, "r") as f:
            dev_kac = f.readlines()
        with open(CONV_ENG, "r") as f:
            dev_eng = f.readlines()
    else:
        raise ValueError("`data` argument only supports `flores` or `conv`.")

    dev_kac = [sent.strip() for sent in dev_kac]
    dev_eng = [sent.strip() for sent in dev_eng]
    
    return {"kac": dev_kac,
            "eng": dev_eng}


def main(args: argparse.Namespace) -> None:
    data = load_dev(args.data)
    kacs = data["kac"]
    engs = data["eng"]

    translator = Translator(igt=args.igt,
                            dictionary=args.dictionary,
                            grammar=args.grammar,
                            structured=args.structured,
                            )

    translations = []
    for i, (kac, eng) in enumerate(zip(kacs, engs)):
        translation = translator.translate(kac)
        print(f"Jinghpaw:    {kac}")
        print(f"English:     {eng}")
        print(f"Translation: {translation}")
        if isinstance(translation, dict):
            # structured output
            # translation: Dict[str (kac), str (eng)]
            translations.append(translation["translation"])
        elif isinstance(translation, str):
            translations.append(translation)
        if args.debug:
            break
        if i == args.num_sentences - 1:
            break
    if args.debug:
        df = pd.DataFrame.from_dict(
            {"kac": [kacs[0]],
             "eng": [engs[0]],
             "translation": translations}
            )
        scores = evaluate(translations, [[engs[0]]])
        print(f"Saving the results to translation_debug.csv...")
        df.to_csv("translation_debug.csv")
        with open("translation_debug_scores.csv", "w") as f:
            f.write(scores["bleu"] + "\n" + scores["chrf"])
    else:
        df = pd.DataFrame.from_dict(
            {"kac": kacs[:args.num_sentences],
             "eng": engs[:args.num_sentences],
             "translation": translations}
        )
        outfile = os.path.join(args.folder, args.outfile) if args.folder is not None else args.outfile
        print(f"Saving the results to {outfile}...")
        df.to_csv(outfile)

        stats_file = outfile[:-4] + "_stats.txt"
        print(f"Saving the stats to {stats_file}...")
        scores = evaluate(translations,
                          [engs[:args.num_sentences]])
        with open(stats_file, "w") as f:
            f.write(scores["bleu"] + "\n" + scores["chrf"])

def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(description="The task is to translate Jinghpaw into English.")
    parser.add_argument("--data",
                        choices=["flores", "conv"],
                        help="Evaluation data.")
    parser.add_argument("-o",
                        "--outfile",
                        type=str,
                        default="translations.csv",
                        help="The path to the file for saving translation results.")
    parser.add_argument("-f",
                        "--folder",
                        type=str,
                        default="results",
                        help="The directory path where the output results will be stored.")
    parser.add_argument("-d",
                        "--dictionary",
                        action="store_true",
                        help="If true, you are using dictionary entry retrieval.")
    parser.add_argument("-g",
                        "--grammar",
                        action="store_true",
                        help="If true, you are using the grammar summary.")
    parser.add_argument("--igt",
                        action="store_true",
                        help="If true, you generate IGT first and then translate; CoT.")
    parser.add_argument("--structured",
                        action="store_true",
                        help="If true, the translator model will generate structured output.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="If true, just one sample is translated."
                        )
    parser.add_argument("-n",
                        "--num_sentences",
                        type=int,
                        default=100,
                        help="The number of sentences to translate.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)    
