# Usage: run from assign2/
#  ./preprocess_scripts/process_backtranslated_data.sh <backtranslated/monoaugment>
#
# Will create new directories:
# data_backtranslated_for_O2M and data_backtranslated_for_M2O
# or
# data_monoaugment_for_O2M and data_monoaugment_for_M2O

if [[ $# -ge 1 && $1 == "monoaugment" ]]; then
    augmentation_method=monoaugment
elif [[ $# -ge 1 && $1 == "mono_and_bt" ]]; then
    augmentation_method=mono_and_bt
elif [[ $# -eq 0  || $1 == "backtranslated" ]]; then
    augmentation_method=backtranslated
elif [[ $# -eq 0  || $1 == "tagged_backtranslated" ]]; then 
    augmentation_method=tagged_backtranslated
elif [[ $# -eq 0  || $1 == "filtered_tagged_backtranslated" ]]; then 
    augmentation_method=filtered_tagged_backtranslated
elif [[ $# -eq 0  || $1 == "dummy_monoaugment" ]]; then 
    augmentation_method=dummy_monoaugment
elif [[ $# -eq 0  || $1 == "dummy_monoaugment_source" ]]; then 
    augmentation_method=dummy_monoaugment_source    
elif [[ $# -eq 1  || $1 == "noisy_monoaugment" ]]; then
    augmentation_method=noisy_monoaugment
    swap_num_pairs=2
    if [[ $# -eq 2 ]]; then
    	swap_num_pairs=$2
    fi
else
    echo "Unknown augmentation method $1 received - give either monolingual (default) or monoaugment"
    exit 1
fi

echo $augmentation_method
backtranslation_data_prefix=data_${augmentation_method}_for

for direction in O2M M2O; do
    output_directory=${backtranslation_data_prefix}_${direction}
    mkdir -p $output_directory
    mkdir -p ${output_directory}/ted_raw
done

for lang in bel aze tur rus kur mar ben; do
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


        echo "creating directory $output_directory"
        backtranslation_directory=data/monolingual_data
        mkdir -p $output_directory

        # Copy dev and test files as is, from "clean" data directory
        cp ${clean_directory}/ted-test.orig.${lang}-eng $output_directory/
        cp ${clean_directory}/ted-dev.orig.${lang}-eng $output_directory/

        training_output_path=$(pwd)/${output_directory}/ted-train.orig.${lang}-eng
        dev_output_path=$(pwd)/${output_directory}/ted-dev.orig.${lang}-eng
        test_output_path=$(pwd)/${output_directory}/ted-test.orig.${lang}-eng

        backtranslated_data=$(pwd)/${backtranslation_directory}/${lang}/${srclang}_bilingual_translated.txt
        clean_target_data=$(pwd)/${backtranslation_directory}/${trglang}/${trglang}_monolingual.txt
        clean_source_data=$(pwd)/${backtranslation_directory}/${srclang}/${srclang}_monolingual.txt
        clean_parallel_corpus_path=$(pwd)/${clean_directory}/ted-train.orig.${lang}-eng

        if [ $augmentation_method = backtranslated ]; then
            # Produce training file, using a concatenation of backtranslated data and clean data
            python preprocess_scripts/process_translation_output.py \
                --output_path $training_output_path \
                --backtranslated_data $backtranslated_data \
                --clean_target_data $clean_target_data \
                --clean_parallel_data_path $clean_parallel_corpus_path \
                --direction ${direction} \
                --backtranslation_augmentation \
                --shuffle_lines
        elif [ $augmentation_method = mono_and_bt ]; then
            # Produce training file, using a concatenation of backtranslated data, monolingual copied data, and clean data
            python preprocess_scripts/process_translation_output.py \
                --output_path $training_output_path \
                --backtranslated_data $backtranslated_data \
                --clean_target_data $clean_target_data \
                --clean_parallel_data_path $clean_parallel_corpus_path \
                --direction ${direction} \
                --backtranslation_augmentation \
                --monolingual_data_augmentation \
                --shuffle_lines
    	elif [ $augmentation_method = noisy_monoaugment ]; then
            # Produce training file, using a concatenation of noisy monolingual data and clean data
            python preprocess_scripts/process_translation_output.py \
            --output_path $training_output_path \
            --clean_target_data $clean_target_data \
            --clean_parallel_data_path $clean_parallel_corpus_path \
            --direction ${direction} \
            --monolingual_noisy_data_augmentation \
            --swap_num_pairs ${swap_num_pairs} \
            --shuffle_lines
        elif [ $augmentation_method = tagged_backtranslated ]; then
            # Produce training file, using a concatenation of backtranslated data (preppended with the tag 'noisy') and clean data (prepended with the tag 'clean')
            python preprocess_scripts/process_translation_output.py \
            --output_path $training_output_path \
            --backtranslated_data $backtranslated_data \
            --clean_target_data $clean_target_data \
            --clean_parallel_data_path $clean_parallel_corpus_path \
            --direction ${direction} \
            --backtranslation_augmentation \
            --tagged_backtranslation \
            --shuffle_lines
        elif [ $augmentation_method = filtered_tagged_backtranslated ]; then
            # Produce training file, using a concatenation of backtranslated data (preppended with the tag 'noisy') and clean data (prepended with the tag 'clean')
            python preprocess_scripts/process_translation_output.py \
            --output_path $training_output_path \
            --backtranslated_data $backtranslated_data \
            --clean_target_data $clean_target_data \
            --clean_parallel_data_path $clean_parallel_corpus_path \
            --direction ${direction} \
            --backtranslation_augmentation \
            --tagged_backtranslation \
            --filtered_tagged \
            --shuffle_lines
        elif [ $augmentation_method = dummy_monoaugment ]; then
            python preprocess_scripts/process_translation_output.py \
                --output_path $training_output_path \
                --clean_target_data $clean_target_data \
                --clean_parallel_data_path $clean_parallel_corpus_path \
                --direction ${direction} \
                --dummy_monoaugmentation \
                --shuffle_lines
        elif [ $augmentation_method = dummy_monoaugment_source ]; then
            python preprocess_scripts/process_translation_output.py \
                --output_path $training_output_path \
                --clean_target_data $clean_target_data \
                --clean_parallel_data_path $clean_parallel_corpus_path \
                --direction ${direction} \
                --dummy_monoaugmentation_source \
                --shuffle_lines
        else python preprocess_scripts/process_translation_output.py \
                --output_path $training_output_path \
                --clean_target_data $clean_target_data \
                --clean_parallel_data_path $clean_parallel_corpus_path \
                --direction ${direction} \
                --monolingual_data_augmentation \
                --shuffle_lines
        fi

    if [[ $augmentation_method = tagged_backtranslated || $augmentation_method = filtered_tagged_backtranslated ]]; then
        if [ $direction = O2M ]; then
            sed -e 's/|||/||| <clean>/' -i $dev_output_path
            sed -e 's/|||/||| <clean>/' -i $test_output_path
        else
            sed -e 's/^/<clean> /' -i $dev_output_path
            sed -e 's/^/<clean> /' -i $test_output_path
        fi
    fi
    done
done
