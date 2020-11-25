# python eval_scripts/comparing_languages_pass_through_rates.py --data_directory data/ted_raw

import argparse
import os
from pass_through_word_accuracy import ParallelTranslation, load_parallel_from_labels, compute_pass_through_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_directory", type=str, default="data/ted_raw", help="Fairseq-generated file of translation decodings")
    args = parser.parse_args()
    if not os.path.isdir(args.data_directory):
        raise ValueError("Data directory provided is not valid. Should be the ted_raw directory from the TED Talks dataset.")

    pass_through_rate_by_language = {}
    for language_pair in os.listdir(args.data_directory):
        language_dir = os.path.join(args.data_directory, language_pair)
        if not os.path.isdir(language_dir):
            continue
        language_pair_corrected = language_pair.replace('_', '-')
        test_corpus_file = os.path.join(language_dir, f"ted-test.orig.{language_pair_corrected}")
        if not os.path.isfile(test_corpus_file):
            print(f"No parallel test corpus found in {test_corpus_file}")
            continue

        parallel_sentences = load_parallel_from_labels(test_corpus_file)
        test_size = len(open(test_corpus_file).read().split("\n"))
        _, _, trg_pass_through_rate = compute_pass_through_words(parallel_sentences)
        pass_through_rate_by_language[language_pair] = (trg_pass_through_rate, test_size)
    
    pass_through_rate_by_language = {k: v for k, v in sorted(pass_through_rate_by_language.items(), key=lambda item: item[1][0])}

    print("Pass-through rates by language:")
    print(f"Lang code\tPass Through Rate\t# of rows in test set")
    for lang, (pass_through_rate, test_size) in pass_through_rate_by_language.items():
        print(f"{lang}\t\t{round(pass_through_rate, 4)}\t\t\t{test_size}")