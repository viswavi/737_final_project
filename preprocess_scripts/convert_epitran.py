import epitran
import re
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="File to transliterate with epitran")
    parser.add_argument('--output_file', type=str, required=True, help="Output from epitran processing")
    parser.add_argument('--first_lang', type=str, default="aze", help="1st language")
    parser.add_argument('--second_lang', type=str, default="spa", help="2nd language")
    args = parser.parse_args()
    return args

args = parse_arguments()

# transliterate both langs to latin script
epi_first = epitran.Epitran('{}-Latn'.format(args.first_lang))
epi_second = epitran.Epitran('{}-Latn'.format(args.second_lang))

with open(args.input_file, "r") as f:
    lines = f.readlines()

with open(args.output_file, "w") as f:
    for line in lines:
        line = re.split("[|]", line)
        first = line[0].strip()
        second = line[-1].strip()
        first = epi_first.transliterate(first)
        second = epi_second.transliterate(second)
        f.write(" ||| ".join([first, second]) + " \n")
