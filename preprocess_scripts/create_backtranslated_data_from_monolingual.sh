# usage: ./create_backtranslated_data <srclang> <trglang> <beam/sampling>
# will then create combined pure and backtranslated src-trg corpus at
# data/ted_raw_backtrans_{M2O/O2M}_{beam/sampling}/
#
# Need to run preprocess_scripts/make-ted-bilingual.sh before you can run this.

sample=1
if [[ $# -ge 1 && $1 -eq "beam" ]]; then
    sample=0
else
    sample=1
fi
if [[ $# -eq 2 ]]; then
    data_size=$2
else
    data_size=6000
fi

sample_or_beam=`if [ $sample -eq 1 ]; then echo "--sampling --beam 1 --nbest 1"; else echo "--beam 5"; fi`

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

        # Randomly sample 6000 sentences from the original set, to reduce computational load
        cp data/monolingual_data/${lang}/${lang}_monolingual_full.txt $processed_monolingual_path
        shuf $processed_monolingual_path -o $processed_monolingual_path
        head -n $data_size $processed_monolingual_path >> data/monolingual_data/${lang}/${lang}_monolingual_copy.txt
        mv data/monolingual_data/${lang}/${lang}_monolingual_copy.txt $processed_monolingual_path
    fi
done

FAIR_SCRIPTS=fairseq/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
TOKENIZER=mosesdecoder/scripts/tokenizer/tokenizer.perl
VOCAB_SIZE=8000


for lrl in aze bel #rus tur
do
    echo "lrl: ${lrl}"
    # Eng -> others is O2M
    # Others -> Eng is M2O
    for style in bilingual #multilingual (fairseq-interactive does not seem to work for multilingual models, at present)
    do
        echo "style: ${style}"
        if [ $style = multilingual ]; then
            if [ $lrl = aze ]; then
                langs="aze tur"
            elif [ $lrl = bel ]; then
                langs="bel rus"
            else
                echo "Error 1!"
                exit 1
            fi
        else
            langs=$lrl
        fi


        for lang in $langs
        do
            echo "lang: $lang"
            # prepare data for this language in this direction
            raw_lang_file=data/monolingual_data/${lang}/${lang}_monolingual.txt
            raw_eng_file=data/monolingual_data/eng/eng_monolingual.txt
            tok_lang_file=${raw_lang_file}.mtok
            tok_eng_file=${raw_eng_file}.mtok

            # Run Moses tokenizer on raw monolingual text, dump to file.
            cat $raw_lang_file | perl $TOKENIZER > $tok_lang_file
            cat $raw_eng_file | perl $TOKENIZER > $tok_eng_file

            if [[ $lang = "tur" && $style = "multilingual" || $lang = "aze" && $style = "multilingual" ]]; then
                lang_spm=azetur
            elif [[ $lang = "rus" && $style = "multilingual"  || $lang = "bel" && $style = "multilingual" ]]; then
                lang_spm=belrus
            elif [[ $lang = "aze" || $lang = "bel" || $lang = "rus" || $lang = "tur" ]]; then
                lang_spm=$lang
            else
                echo "Error 2"
                exit 1
            fi

            if [ $style = bilingual ]; then
                style_suffix=spm
            elif [ $style = multilingual ]; then
                style_suffix=sepspm
            else
                echo "Error 3"
                exit 1
            fi

            model_dir=${lang_spm}_${style_suffix}${VOCAB_SIZE}
            PROC_DDIR=data/ted_processed/${model_dir}/
            bpe_lang_file=${raw_lang_file}.bpe
            bpe_eng_file=${raw_eng_file}.bpe

            # Run BPE for data
            python "$SPM_ENCODE" \
                --model="$PROC_DDIR"/${lang}_eng/spm"$VOCAB_SIZE".model \
                --output_format=piece \
                --inputs $tok_lang_file $tok_eng_file \
                --outputs $bpe_lang_file $bpe_eng_file

            for direction in O2M M2O
            do
                # TODO(Vijay): delete the below chunk
                if [[ $lang = aze && $direction = O2M ]]; then
                    continue
                fi
                # TODO(Vijay): delete the above chunk
                printf "\n$lang $direction\n"

                if [ $direction = O2M ]; then
                    srclang=eng
                    trglang=$lang
                    bpe_path=${bpe_eng_file}
                else
                    srclang=${lang}
                    trglang=eng
                    bpe_path=${bpe_lang_file}
                fi

                if [ $style = bilingual ]; then
                    if [ $direction = O2M ]; then
                        temp=eng_${lang}
                    else
                        temp=${lang}_eng
                    fi
                    direction=$temp
                else
                    # Construct multilingual-specific parameters
                    if [ $direction = M2O ]; then
                        encoder_langtok="src"
                        if [ $lrl = aze ]; then
                            lang_pairs="aze-eng,tur-eng"
                        else
                            lang_pairs="bel-eng,rus-eng"
                        fi
                    elif [ $direction = O2M ]; then
                        encoder_langtok="tgt"
                        if [ $lrl = aze ]; then
                            lang_pairs="eng-aze,eng-tur"
                        else
                            lang_pairs="eng-bel,eng-rus"
                        fi
                    else
                        echo "Error 4"
                        exit 1
                    fi
                fi

                if [ $style = multilingual ]; then
                    multilingual_parameters="--task translation_multi_simple_epoch --lang-pairs ${lang_pairs} --encoder-langtok ${encoder_langtok} --source-lang ${srclang} --target-lang ${trglang}"
                else
                    multilingual_parameters=
                fi

                fairseq-interactive \
                    --path fairseq/checkpoints/ted_${model_dir}/${direction}/checkpoint_best.pt \
                    --batch-size 32 \
                    --buffer-size 32 \
                    --tokenizer moses \
                    --remove-bpe sentencepiece \
                    --scoring sacrebleu \
                    ${sample_or_beam} ${multilingual_parameters} \
                    --input $bpe_path \
                     fairseq/data-bin/ted_${model_dir}/${direction} > data/monolingual_data/${lang}/${trglang}_${style}_translated.txt
            done
        done
    done
done

