# Usage: run from assign2/
#  ./preprocess_scripts/process_backtranslated_data.sh <backtranslated/monoaugment>
#
# Will create new directories:
# data_backtranslated_for_O2M and data_backtranslated_for_M2O
# or
# data_monoaugment_for_O2M and data_monoaugment_for_M2O

if [[ $# -ge 1 && $1 -eq "monoaugment" ]]; then
    augmentation_method=monoaugment
elif [[ $# -eq 0  || $1 -ne "backtranslated" ]]; then
    augmentation_method=backtranslated
else
    echo "Unknown augmentation method $1 received - give either monolingual (default) or monoaugment"
    exit 1
fi

backtranslation_data_prefix=data_${augmentation_method}_for

for direction in O2M M2O; do
    output_directory=${backtranslation_data_prefix}_${direction}
    mkdir -p $output_directory
    mkdir -p ${output_directory}/ted_raw
done

for lang in bel aze tur rus; do
    for direction in O2M M2O; do
        if [ $direction = O2M ]; then
            srclang=eng
            trglang=$lang
        else
            srclang=${lang}
            trglang=eng
        fi

        clean_directory=data/ted_raw/${lang}_eng
        output_directory=${backtranslation_data_prefix}_${direction}/ted_raw/${lang}_eng
        backtranslation_directory=data/monolingual_data
        mkdir -p $output_directory

        # Copy dev and test files as is, from "clean" data directory
        cp ${clean_directory}/ted-test.orig.${lang}-eng $output_directory/
        cp ${clean_directory}/ted-dev.orig.${lang}-eng $output_directory/

        training_output_path=$(pwd)/${output_directory}/ted-train.orig.${lang}-eng
        backtranslated_data=$(pwd)/${backtranslation_directory}/${lang}/${srclang}_bilingual_translated.txt
        clean_target_data=$(pwd)/${backtranslation_directory}/${trglang}/${trglang}_monolingual.txt
        clean_parallel_corpus_path=$(pwd)/${clean_directory}/ted-train.orig.${lang}-eng

        if [ $augmentation_method = backtranslated ]; then
            # Produce training file, using a concatenation of backtranslated data and clean data
            python preprocess_scripts/process_translation_output.py \
                --output_path $training_output_path \
                --backtranslated_data $backtranslated_data \
                --clean_target_data $clean_target_data \
                --clean_parallel_data_path $clean_parallel_corpus_path \
                --shuffle_lines
        else
            python preprocess_scripts/process_translation_output.py \
                --output_path $training_output_path \
                --clean_target_data $clean_target_data \
                --clean_parallel_data_path $clean_parallel_corpus_path \
                --monolingual_data_augmentation \
                --shuffle_lines
        fi
    done
done
        
    