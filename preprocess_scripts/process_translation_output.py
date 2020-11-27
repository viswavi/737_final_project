#!/usr/bin/env python
#
# This script consumes the stdout dumped from running translation for a file, and
# produces data in the raw corpus format, with translated "target" as backtranslated "source".
# It can alternatively take in pure monolingual data, treating the monolingual target data
# as both source and target.
#
## To generate new parallel corpus augmented with backtranslated data:
# python process_translation_output.py --output_path <filepath> \
#     --clean_target_data <path to true target data for backtranslations
#     --backtranslated_data <path to backtranslated source data> \
#     --clean_parallel_data_path <path to human-labeled, clean parallel corpus> \
#     --shuffle_lines
#
## To generate new parallel corpus augmented with monolingual data copying:
# python process_translation_output.py --output_path <filepath> \
#     --clean_target_data <path to true target data for backtranslations
#     --clean_parallel_data_path <path to human-labeled, clean parallel corpus> \
#     --monolingual_data_augmentation \
#     --shuffle_lines
#
import argparse
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description=(
        # Adapted from fairseq/examples/backtranslation/extract_bt_data.py
        'Extract back-translations from the stdout of fairseq-generate.'
    ))
    parser.add_argument('--output_path', '-o', required=True, 
        help='File to store processed backtranslation data (in the raw corpus format)')
    parser.add_argument('--backtranslated_data', '-i',
        help='File containing the stdout dumped from running translation')
    parser.add_argument('--clean_parallel_data_path', '-f',
        help='Clean "target" text data to pair with backtranslated "source"')
    parser.add_argument('--clean_target_data', '-clean',
        help='Whether to combine backtranslated data with original target-to-source data')
    parser.add_argument('--backtranslation_augmentation', '-bt',
        help='Perform data augmentation via backtranslation', action='store_true')
    parser.add_argument('--monolingual_data_augmentation', '-mono',
        help='If true, copy target data as "source" data', action='store_true')
    parser.add_argument('--shuffle_lines', action='store_true',
        help='Whether or not to shuffle lines of data')
    args = parser.parse_args()


    clean_target_data = open(args.clean_target_data).read().split("\n")
    outlines = []

    if args.backtranslation_augmentation:
        backtranslated_output = open(args.backtranslated_data).read().split("\n")
        for line in tqdm(backtranslated_output):
            if len(line) <= 6:
                continue
            line_type = line[0]
            if line_type != "D":
                continue

            target_line = line.split('\t')[2]
            line_number = int(line.split('\t')[0][2:])
            source_line = clean_target_data[line_number]
            outlines.append(f"{source_line} ||| {target_line}")
    if args.monolingual_data_augmentation:
        for source_line in clean_target_data:
            if len(source_line.split()) > 0:
                # skipping empty lines, add monolingual data as source and target
                outlines.append(f"{source_line} ||| {source_line}")

    if args.clean_parallel_data_path is not None:
        clean_data_lines = open(args.clean_parallel_data_path).read().split("\n")
        # Drop final row, which is empty
        clean_data_lines = clean_data_lines[:-1]
        outlines.extend(clean_data_lines)

    if args.shuffle_lines:
        random.shuffle(outlines)

    outfile = open(args.output_path, 'w')
    outfile.write("\n".join(outlines))
    outfile.close()

if __name__ == "__main__":
    main()
