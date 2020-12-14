#!/usr/bin/env python
#
# This script consumes the stdout dumped from running translation for a file, and
# produces data in the raw corpus format, with translated "target" as backtranslated "source".
# It can alternatively take in pure monolingual data, treating the monolingual target data
# as both source and target. In this case, it uses the "mono_to_parallel_ratio" argument
# to decide how much monolingual data to use, relative to parallel data.
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
import itertools

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
    parser.add_argument('--clean_target_data', '--clean_target',
        help='Monolingual data on the target side to use for data augmentation')
    parser.add_argument('--clean_source_data', '--clean_source',
        help='Monolingual data on the source side to use for data augmentation')
    parser.add_argument('--backtranslation_augmentation', '-bt',
        help='Perform data augmentation via backtranslation', action='store_true')
    parser.add_argument('--monolingual_data_augmentation', '-mono_target',
        help='If true, copy target data as "source" data', action='store_true')
    parser.add_argument('--source_data_augmentation', '-mono_source',
        help='If true, copy source data as "target" data', action='store_true')
    parser.add_argument('--monolingual_noisy_data_augmentation', '-mono_noise',
        help='If true, copy target data as "source" data and swap n pair of words', action='store_true')
    parser.add_argument('--swap_num_pairs', '-n', type=int, default=1, help='Num of pairs to swap, if -mono_noise is true')
    parser.add_argument('--monolingual_masked_data_augmentation', '-mono_masked', help='If true, copy target data as "source" data and mask n words', action='store_true')
    parser.add_argument('--num_masks', '-m', type=int, default=1, help='Num of words to mask, if -mono_masked is true')
    parser.add_argument('--tagged_backtranslation', '-tagged', 
        help='If true, training data will be prepended with noisy or clean', action='store_true')
    parser.add_argument('--filtered_tagged', '-filtered', 
        help='If true, training data will be filtered', action='store_true')   
    parser.add_argument('--dummy_monoaugmentation', '-dummy_mono_target',
        help='If true, copy target data as "source" data', action='store_true')
    parser.add_argument('--dummy_monoaugmentation_source', '-dummy_mono_source_target',
        help='If true, copy target data as "source" data', action='store_true')
    parser.add_argument('--shuffle_lines', action='store_true',
        help='Whether or not to shuffle lines of data')
    parser.add_argument('--mono_to_parallel_ratio', type=float, default=2.0,
        help='Ratio of monolingual data to parallel data to use')
    args = parser.parse_args()

    parallel_data_size = len(open(args.clean_parallel_data_path).read().split("\n")) - 1
    monolingual_data_size = int(args.mono_to_parallel_ratio * parallel_data_size)

    clean_target_data = open(args.clean_target_data).read().split("\n")
    outlines = []

    if args.backtranslation_augmentation:
        backtranslation_lines = []
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
                    backtranslation_lines.append(f"{clean_target_line} ||| <noisy> {backtranslated_sentence}")
                else:
                    backtranslation_lines.append(f"{clean_target_line} ||| {backtranslated_sentence}")
            elif args.direction == "M2O":
                if args.tagged_backtranslation:
                    backtranslation_lines.append(f"<noisy> {backtranslated_sentence} ||| {clean_target_line}")
                else:
                    backtranslation_lines.append(f"{backtranslated_sentence} ||| {clean_target_line}")
            else:
                raise ValueError("Direction should be O2M or M2O")
        outlines.extend(backtranslation_lines[:monolingual_data_size])
    
    if args.monolingual_data_augmentation:
        copied_lines = []
        for target_line in clean_target_data:
            if len(target_line.split()) > 0:
                # skipping empty lines, add monolingual data as source and target
                copied_lines.append(f"{target_line} ||| {target_line}")
        outlines.extend(copied_lines[:monolingual_data_size])


    if args.monolingual_noisy_data_augmentation:
        noised_copied_data = []
        for source_line in clean_target_data:
            if len(source_line.split()) > 0:
                # skipping empty lines
                # add noisy monolingual data as source and monolingual data as target
                noisy_line = add_noise(source_line, args.swap_num_pairs)
                noised_copied_data.append(f"{noisy_line} ||| {source_line}")
        outlines.extend(noised_copied_data[:monolingual_data_size])
    
    if args.monolingual_masked_data_augmentation:
        masked_copied_data = []
        for source_line in clean_target_data:
            if len(source_line.split()) > 0:
                # skipping empty lines
                # add masked monolingual data as source and monolingual data as target
                masked_line = add_masking(source_line, args.num_masks)
                masked_copied_data.append(f"{masked_line} ||| {source_line}")
        outlines.extend(masked_copied_data[:monolingual_data_size])

    if args.dummy_monoaugmentation:
        ind = 0
        dummy_words = get_dummy_words(4)
        copied_lines = []
        for target_line in clean_target_data:
            if len(target_line.split()) > 0:
                dummy_line = ' '.join(map(str,dummy_words[ind:ind+len(target_line.split())]))
                ind += len(target_line.split())
                # skipping empty lines, add monolingual data as source and target
                copied_lines.append(f"{dummy_line} ||| {dummy_line}")
        outlines.extend(copied_lines[:monolingual_data_size])

    if args.dummy_monoaugmentation_source:
        ind = 0
        dummy_words = get_dummy_words(4)
        copied_lines = []
        for target_line in clean_target_data:
            if len(target_line.split()) > 0:
                dummy_line = ' '.join(map(str,dummy_words[ind:ind+len(target_line.split())]))
                ind += len(target_line.split())
                # skipping empty lines, add monolingual data as source and target
                if args.direction == "O2M":
                    copied_lines.append(f"{target_line} ||| {dummy_line}")
                else:
                    copied_lines.append(f"{dummy_line} ||| {target_line}")
        outlines.extend(copied_lines[:monolingual_data_size])

    if args.filtered_tagged:
        outlines = list(filter(lambda line: check_keep(line, "noisy"), outlines))

    if args.clean_parallel_data_path is not None:
        clean_data_lines = open(args.clean_parallel_data_path).read().split("\n")
        # Drop final row, which is empty
        clean_data_lines = clean_data_lines[:-1]
        if args.tagged_backtranslation:
            if args.direction == 'O2M':
                clean_data_lines = [i.replace('|||', '||| <clean>') for i in clean_data_lines]
            else:
                clean_data_lines = ['<clean> %s' % i for i in clean_data_lines]
            if(args.filtered_tagged):
                clean_data_lines = list(filter(lambda line: check_keep(line, "clean"), clean_data_lines))

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

def add_masking(source_line, num_masks):
    source_line = source_line.split()
    l = len(source_line)
    if num_masks > (l//2):
        num_masks = l//2
    masked_line = source_line.copy()
    masked_line = set_mask(masked_line, num_masks, l)
    return " ".join(masked_line)

def set_mask(line, num_masks, l):
    idxs = random.sample(range(0, l-1), num_masks)
    for idx in idxs:
        line[idx] = "<**MASK**>"
    return line

def check_keep(line, tag):
    sentences = line.split("|||")
    res = True
    if(tag == "clean" and (len(sentences[0].split(" ")) > 251 or len(sentences[1].split(" ")) > 251)):
        res = False
    if(tag == "noisy" and (len(sentences[0].split(" ")) > 76 or len(sentences[1].split(" ")) > 76)):
        res = False

    return res

def get_dummy_words(length):
    characters = [u'\u0b95', u'\u0b99', u'\u0b9a', u'\u0b9e', u'\u0b9f', u'\u0ba3', u'\u0ba4', u'\u0ba8', u'\u0baa', u'\u0ba4', u'\u0bae', u'\u0baf', u'\u0bb0', u'\u0bb2', u'\u0bb5', u'\u0bb4',  u'\u0bb3', u'\u0bb1',  u'\u0ba9', u'\u0b85', u'\u0b86', u'\u0b87', u'\u0b88', u'\u0b89', u'\u0b8A', u'\u0b8E', u'\u0b8F', u'\u0b90', u'\u0b92', u'\u0b93', u'\u0b94']
    words = []
    for i in itertools.product(characters, repeat=length):
        words.append(''.join(map(str, i)))
    return words

if __name__ == "__main__":
    main()
