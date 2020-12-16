# Example usage:
# python eval_scripts/pass_through_word_accuracy.py \
# --decoding_file fairseq/checkpoints_monoaugment_for_M2O/ted_aze_spm8000/aze_eng/test_b5.log 

import argparse
from string import punctuation
from tqdm import tqdm

from pass_through_word_accuracy import ParallelTranslation, load_parallel_from_decodings, load_parallel_from_labels

def tokenize(sentence):
    return sentence.split()



def compute_sentence_lengths(parallel_sentences):
    label_lengths = []
    predicted_lengths = []
    for pair in parallel_sentences.values():
        target_labeled = pair.target_labeled
        target_predicted = pair.target_predicted

        target_labeled_tokens = tokenize(target_labeled)
        target_predicted_tokens = tokenize(target_predicted)

        label_lengths.append(len(target_labeled_tokens))
        predicted_lengths.append(len(target_predicted_tokens))

    avg_label_length = float(sum(label_lengths)) / len(label_lengths)
    avg_predicted_length = float(sum(predicted_lengths)) / len(predicted_lengths)
    return avg_label_length, avg_predicted_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--decoding_file", type=str, required=True, help="Fairseq-generated file of translation decodings")
    args = parser.parse_args()

    parallel_sentences = load_parallel_from_decodings(args.decoding_file)
    avg_label_length, avg_predicted_length = compute_sentence_lengths(parallel_sentences)
    print(f"Average GT Length: {avg_label_length}")
    print(f"Average Predicted Length: {avg_predicted_length}")
