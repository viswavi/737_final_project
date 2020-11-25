# Example usage:
# python eval_scripts/find_pass_through_words.py \
# --decoding_file fairseq/checkpoints_monoaugment_for_M2O/ted_aze_spm8000/aze_eng/test_b5.log 

import argparse
from string import punctuation
from tqdm import tqdm


class ParallelTranslation:
    def __init__(self, source_sentence=None, target_labeled=None, target_predicted=None):
        self.source_sentence = source_sentence
        self.target_labeled = target_labeled
        self.target_predicted = target_predicted

    def __str__(self):
        return str({"source_sentence": self.source_sentence,
                    "target_labeled": self.target_labeled,
                    "target_predicted": self.target_predicted})


def load_parallel(decoding_file):
    parallel_sentences = {}
    backtranslated_output = open(decoding_file).read().split("\n")
    for line in tqdm(backtranslated_output):
        if not line.startswith("S-") and not line.startswith("T-") and not line.startswith("D-"):
            continue
        line_type = line[0]
        line_number = int(line.split('\t')[0][2:])

        if line_type is "S":
            source_sentence = line.split('\t')[1]
            parallel_sentences[line_number] = ParallelTranslation(source_sentence = source_sentence)
        elif line_type is "T":
            target_labeled = line.split('\t')[1]
            parallel_sentences[line_number].target_labeled = target_labeled
        elif line_type is "D":
            target_predicted = line.split('\t')[2]
            parallel_sentences[line_number].target_predicted = target_predicted
        else:
            raise ValueError("Unexpected line {line} found")

    # validate data
    for v in parallel_sentences.values():
        if v.source_sentence is None or v.target_labeled is None or v.target_predicted is None:
            raise ValueError("All lines should be fully processed by this point - indicates an issue in decoding")

    return parallel_sentences


def clean_tokenize(sentence):
    '''
    As Currey et al do, we compute tokens in lowercase, excluding words that only contain one
    character.
    Unlike them, we also strip trailing punctuation from the sentence, (e.g. a period
    attached to the final word in the sentence), which seems important for a fair evaluation.
    '''
    sentence_lower = sentence.lower()
    sentence_cleaned = sentence_lower.strip(punctuation)
    sentence_tokens = sentence_cleaned.split()
    sentence_tokens = [t for t in sentence_tokens if len(t) > 1]
    return set(sentence_tokens)


def compute_pass_through_words(parallel_sentences):
    pass_through_words = {}
    total_number_of_source_words = 0
    total_number_of_target_words = 0
    total_number_of_pass_throughs = 0
    for line_number, pair in parallel_sentences.items():
        source_labeled = pair.source_sentence
        target_labeled = pair.target_labeled

        source_tokens = clean_tokenize(source_labeled)
        target_tokens = clean_tokenize(target_labeled)

        pass_throughs = target_tokens.intersection(source_tokens)
        pass_through_words[line_number] = pass_throughs

        total_number_of_source_words += len(source_tokens)
        total_number_of_target_words += len(target_tokens)
        total_number_of_pass_throughs += len(pass_throughs)

    src_pass_through_rate = float(total_number_of_pass_throughs) / total_number_of_source_words
    trg_pass_through_rate = float(total_number_of_pass_throughs) / total_number_of_target_words
    return pass_through_words, src_pass_through_rate, trg_pass_through_rate


def measure_pass_through_prediction_rate(parallel_sentences, pass_through_words):
    total_pass_throughs_labeled = 0
    correct_pass_through_predictions = 0
    total_pass_through_predictions = 0

    for line_number, pair in parallel_sentences.items():
        pass_through_words_labeled = pass_through_words[line_number]

        source_labeled = pair.source_sentence
        target_predicted = pair.target_predicted
        source_tokens = clean_tokenize(source_labeled)
        target_predicted_tokens = clean_tokenize(target_predicted)

        pass_through_words_correctly_predicted = set(target_predicted_tokens).intersection(pass_through_words_labeled)
        all_pass_through_words_predicted = set(target_predicted_tokens).intersection(set(source_tokens))

        correct_pass_through_predictions += len(pass_through_words_correctly_predicted)
        total_pass_throughs_labeled += len(pass_through_words_labeled)
        total_pass_through_predictions += len(all_pass_through_words_predicted)

    pass_through_recall  = float(correct_pass_through_predictions) / total_pass_throughs_labeled
    print(f"Pass through recall: {pass_through_recall}")
    if total_pass_through_predictions is 0:
        print("No pass through predictions made (precision is trivially 1.0)")
    else:
        pass_through_precision  = float(correct_pass_through_predictions) / total_pass_through_predictions
        print(f"Pass through precision: {pass_through_precision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--decoding_file", type=str, required=True, help="Fairseq-generated file of translation decodings")
    args = parser.parse_args()
    parallel_sentences = load_parallel(args.decoding_file)
    pass_through_words, src_pass_through_rate, trg_pass_through_rate = compute_pass_through_words(parallel_sentences)
    print(f"Source pass-through word rate: {src_pass_through_rate}")
    print(f"Target pass-through word rate: {trg_pass_through_rate}")

    measure_pass_through_prediction_rate(parallel_sentences, pass_through_words)