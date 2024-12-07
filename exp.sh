echo "Running baseline"
poetry run python evaluate.py --structured --outfile dev100_baseline.csv
echo "Running IGT"
poetry run python evaluate.py --structured --outfile dev100_igt.csv --igt
echo "Running Dictionary"
poetry run python evaluate.py --structured --outfile dev100_dict.csv --dictionary
echo "Running Grammar"
poetry run python evaluate.py --structured --outfile dev100_grammar.csv --grammar
echo "Running Dictionary, Grammar"
poetry run python evaluate.py --structured --outfile dev_100_dict_grammar.csv --dictionary --grammar
echo "Running IGT, Dictionary"
poetry run python evaluate.py --structured --outfile dev100_igt_dict.csv --igt --dictionary
echo "Running IGT, Grammar"
poetry run python evaluate.py --structured --outfile dev100_igt_grammar.csv --igt --grammar
echo "Running IGT, Dictionary, Grammar"
poetry run python evaluate.py --structured --outfile dev100_igt_dict_grammar.csv --igt --dictionary --grammar

