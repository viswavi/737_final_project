#!/usr/bin/env python
# python prepare_monolingual_data.py --input_file <filepath> --output_path <newpath>
# This script consumes raw monolingual corpus sentences from https://wortschatz.uni-leipzig.de/en/download/, 
# and produces data in a format accepted by fairseq-generate.

'''
python preprocess_scripts/prepare_monolingual_data_from_parallel.py --input_file \
/home/vijay/11-737-group/assign2/data/monolingual_data/aze_newscrawl_2013_30K/aze_newscrawl_2013_30K-sentences.txt \
--output_path /home/vijay/11-737-group/assign2/data/monolingual_data/aze.txt
'''


import argparse
from tqdm import tqdm

def process_line(line):
    if len(line.split('\t')) > 1:
        return line.split('\t')[1]
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description=(
        'Consume monolingual data, and process into fairseq-friendly format'
    ))
    parser.add_argument('--input_file', '-i', required=True, 
        help='File containing raw monolingual data')
    parser.add_argument('--output_path', '-o', required=True, 
        help='File to store processed, fairseq-friendly monlingual data')

    args = parser.parse_args()

    generated_translation_lines = open(args.input_file).read().split("\n")
    outlines = []
    for line in tqdm(generated_translation_lines):
        preprocessed_line = process_line(line)
        if preprocessed_line is not None:
            outlines.append(preprocessed_line)
    
    outfile = open(args.output_path, 'w')
    outfile.write("\n".join(outlines))
    outfile.close()

if __name__ == "__main__":
    main()
