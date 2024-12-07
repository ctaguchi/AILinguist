echo "Running baseline"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_baseline.csv
echo "Running IGT"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_igt.csv --igt
echo "Running Dictionary"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_dict.csv --dictionary
echo "Running Grammar"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_grammar.csv --grammar
echo "Running Dictionary, Grammar"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_dict_grammar.csv --dictionary --grammar
echo "Running IGT, Dictionary"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_igt_dict.csv --igt --dictionary
echo "Running IGT, Grammar"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_igt_grammar.csv --igt --grammar
echo "Running IGT, Dictionary, Grammar"
poetry run python evaluate.py --data conv --structured --outfile dev100_conv_igt_dict_grammar.csv --igt --dictionary --grammar

