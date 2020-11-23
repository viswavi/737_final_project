if [[ ! -e data/monolingual_data/aze/aze_newscrawl_2013_30K/aze_newscrawl_2013_30K-sentences.txt ]]; then
    printf "\nNeed to download monolingual data files and unzip into the right directory structure."
    printf "See https://github.com/viswavi/11-737-group/tree/master/assign2#download-monolingual-data-for-backtranslation--- for the data link."
    printf "Check that your directory structure is correct with \"ls data/monolingual_data/tur/tur_newscrawl_2018_30K/tur_newscrawl_2018_30K-sentences.txt\""
fi

data_size=15000
for i in "aze data/monolingual_data/aze/aze_newscrawl_2013_30K/aze_newscrawl_2013_30K-sentences.txt" "bel data/monolingual_data/bel/bel_newscrawl_2017_30K/bel_newscrawl_2017_30K-sentences.txt" "rus data/monolingual_data/rus/rus_newscrawl-public_2018_30K/rus_newscrawl-public_2018_30K-sentences.txt" "eng data/monolingual_data/eng/eng_newscrawl-public_2018_30K/eng_newscrawl-public_2018_30K-sentences.txt" "tur data/monolingual_data/tur/tur_newscrawl_2018_30K/tur_newscrawl_2018_30K-sentences.txt"
do
    set -- $i
    lang=$1
    raw_monolingual_path=$2
    processed_monolingual_path=data/monolingual_data/${lang}/${lang}_monolingual.txt

    if [[ ! -e $processed_monolingual_path ]]; then
        python preprocess_scripts/prepare_monolingual_data.py --input_file \
        $raw_monolingual_path \
        --output_path data/monolingual_data/${lang}/${lang}_monolingual_full.txt

        # Randomly sample $data_size sentences from the original set, to reduce computational load
        cp data/monolingual_data/${lang}/${lang}_monolingual_full.txt $processed_monolingual_path
        shuf $processed_monolingual_path -o $processed_monolingual_path
        head -n $data_size $processed_monolingual_path >> data/monolingual_data/${lang}/${lang}_monolingual_copy.txt
        mv data/monolingual_data/${lang}/${lang}_monolingual_copy.txt $processed_monolingual_path
    fi
done


printf "\n\nPrepping bilingual data for monoaugment, Aze-Eng\n\n"
./preprocess_scripts/make-ted-bilingual.sh aze monoaugment_for_M2O
printf "\n\nPrepping bilingual data for monoaugment, Eng-Aze\n\n"
./preprocess_scripts/make-ted-bilingual.sh aze monoaugment_for_O2M
printf "\n\nPrepping bilingual data for monoaugment, Bel-Eng\n\n"
./preprocess_scripts/make-ted-bilingual.sh bel monoaugment_for_M2O
printf "\n\nPrepping bilingual data for monoaugment, Eng-Bel\n\n"
./preprocess_scripts/make-ted-bilingual.sh bel monoaugment_for_O2M

printf "\n\nTraining bilingual model for Aze-Eng\n\n"
./job_scripts/monolingual_trainer.sh eng bel monoaugment_for_O2M
printf "\n\nTraining bilingual model for Eng-Aze\n\n"
./job_scripts/monolingual_trainer.sh bel eng monoaugment_for_M2O
printf "\n\nTraining bilingual model for Bel-Eng\n\n"
./job_scripts/monolingual_trainer.sh eng aze monoaugment_for_O2M
printf "\n\nTraining bilingual model for Eng-Bel\n\n"
./job_scripts/monolingual_trainer.sh aze eng monoaugment_for_M2O

# ./preprocess_scripts/make-ted-multilingual.sh aze tur monoaugment_for_O2M
# ./preprocess_scripts/make-ted-multilingual.sh aze tur monoaugment_for_M2O
# ./preprocess_scripts/make-ted-multilingual.sh bel rus monoaugment_for_O2M
# ./preprocess_scripts/make-ted-multilingual.sh bel rus monoaugment_for_M2O

# ./job_scripts/multilingual_trainer.sh eng rus bel monoaugment_for_O2M
# ./job_scripts/multilingual_trainer.sh bel rus eng monoaugment_for_M2O
# ./job_scripts/multilingual_trainer.sh eng tur aze monoaugment_for_O2M
# ./job_scripts/multilingual_trainer.sh aze tur eng monoaugment_for_M2O
