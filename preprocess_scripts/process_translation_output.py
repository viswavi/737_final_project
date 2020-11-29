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
import re

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
    parser.add_argument('--direction', required=True,
        help='M2O (training foreign-to-English model) or O2M (English-to-foreign)')
    parser.add_argument('--clean_target_data', '-clean',
        help='Whether to combine backtranslated data with original target-to-source data')
    parser.add_argument('--backtranslation_augmentation', '-bt',
        help='Perform data augmentation via backtranslation', action='store_true')
    parser.add_argument('--monolingual_data_augmentation', '-mono',
        help='If true, copy target data as "source" data', action='store_true')
    parser.add_argument('--monolingual_noisy_data_augmentation', '-mono_noise',
        help='If true, copy target data as "source" data and swap n pair of words', action='store_true')
    parser.add_argument('--swap_num_pairs', '-n', type=int, default=1, help='Num of pairs to swap, if -mono_noise is true')
    parser.add_argument('--tagged_backtranslation', '-tagged', 
        help='If true, training data will be prepended with noisy or clean', action='store_true')
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

            backtranslated_sentence = line.split('\t')[2]
            line_number = int(line.split('\t')[0][2:])
            clean_target_line = clean_target_data[line_number]
            if args.direction == "O2M":
                if args.tagged_backtranslation:
                    outlines.append(f"{clean_target_line} ||| <noisy> {backtranslated_sentence}")
                else:
                    outlines.append(f"{clean_target_line} ||| {backtranslated_sentence}")
            elif args.direction == "M2O":
                if args.tagged_backtranslation:
                    outlines.append(f"<noisy> {backtranslated_sentence} ||| {clean_target_line}")
                else:
                    outlines.append(f"{backtranslated_sentence} ||| {clean_target_line}")
            else:
                raise ValueError("Direction should be O2M or M2O")
    if args.monolingual_data_augmentation:
        for source_line in clean_target_data:
            if len(source_line.split()) > 0:
                # skipping empty lines, add monolingual data as source and target
                outlines.append(f"{source_line} ||| {source_line}")

    if args.monolingual_noisy_data_augmentation:
        for source_line in clean_target_data:
            if len(source_line.split()) > 0:
                # skipping empty lines
                # add noisy monolingual data as source and monolingual data as target
                noisy_line = add_noise(source_line, args.swap_num_pairs)
                outlines.append(f"{noisy_line} ||| {source_line}")

    if args.tagged_backtranslation:
        # Add noisy label to backtranslated data
        outlines = ['noisy %s' % i for i in outlines]
        # Add clean labels to dev and test

    if args.clean_parallel_data_path is not None:
        clean_data_lines = open(args.clean_parallel_data_path).read().split("\n")
        # Drop final row, which is empty
        clean_data_lines = clean_data_lines[:-1]
        if args.tagged_backtranslation:
            if args.direction == 'O2M':
                clean_data_lines = [i.replace('|||', '||| <clean>') for i in clean_data_lines]
            else:
                clean_data_lines = ['<clean> %s' % i for i in clean_data_lines]
        outlines.extend(clean_data_lines)

    if args.shuffle_lines:
        random.shuffle(outlines)

    outfile = open(args.output_path, 'w')
    outfile.write("\n".join(outlines))
    outfile.close()

def add_noise(source_line, num_swaps):
    source_line = source_line.split()
    l = len(source_line)
    noisy_line = source_line.copy()
    for _ in range(num_swaps):
        noisy_line = swap_word(noisy_line, l)
    return " ".join(noisy_line)

def swap_word(line, l):
    random_idx_1 = random.randint(0, l-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, l-1)
        counter += 1
        # try 3 times to swap
        if counter > 3:
            return line
    line[random_idx_1], line[random_idx_2] = line[random_idx_2], line[random_idx_1]
    return line

if __name__ == "__main__":
    main()
