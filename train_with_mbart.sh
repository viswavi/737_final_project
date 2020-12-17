#!/bin/bash
# This script was adapted from https://github.com/pytorch/fairseq/tree/master/examples/mbart
# It makes it easy to finetune mBART on some language in the Ted Talks dataset (though only 25 languages are
# currently supported by mBART - read the excellent paper for more details: https://arxiv.org/pdf/2001.08210.pdf).
# In particular, this script makes it easy to experiment with finetuning mBART with data augmentation techniques.
#
# Before running this script, you must have prepared monolingual data in the bilingual corpus binary format for
# the mBART-supported language pair of your choice.

if [ $# -ge 2 ]; then
    srclang=$1
    trglang=$2
    if [[ "$srclang" = "eng" || "$srclang" = "en_XX" ]]; then
        lrlang=$trglang
    else
        lrlang=$srclang
    fi

	if [ $# -ge 3 ]; then
	    model_data_suffix=_$3
	else
        model_data_suffix=
    fi

	model_directory_suffix=_mbart${model_data_suffix}

	if [ $# -eq 4 ]; then
	    num_epochs=$4
	else
	    num_epochs=60
	fi
else
    echo 'Error: Must provide a source language code (e.g. aze or bel) 
    and a target language code (e.g. eng)'
    exit 1
fi

DATA_DIR=fairseq/data-bin${model_data_suffix}/ted_${lrlang}_spm8000/${srclang}_${trglang}/
MODEL_DIR=fairseq/checkpoints${model_directory_suffix}/ted_${lrlang}_spm8000/${srclang}_${trglang}/
mkdir -p $MODEL_DIR
GPU_DEVICE=0

printf "\n\nMODEL_DIR: ${MODEL_DIR}\n\n"


# Set the path to your unzipped pretrained mBART model directory.
PRETRAIN=~/monolingual_data_copying/mbart.cc25.v2 # fix if you moved the downloaded checkpoint
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

fairseq-train $DATA_DIR \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang $srclang --target-lang $trglang \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 5 --save-interval-updates 999999 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2 \
  --save-dir $MODEL_DIR \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $langs \
  --max-epoch $num_epochs \
  --ddp-backend no_c10d \
  --fp16

fairseq-generate $DATA_DIR \
  --path $MODEL_DIR/checkpoint_best.pt \
  --task translation_from_pretrained_bart \
  --gen-subset valid \
  -s $srclang -t $trglang \
  --tokenizer moses \
  --remove-bpe sentencepiece \
  --scoring sacrebleu \
  --batch-size 1 --langs $langs --beam 5  > "$MODEL_DIR"/valid_b5.log

fairseq-generate $DATA_DIR \
  --path   $MODEL_DIR/checkpoint_best.pt \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -s $srclang -t $trglang \
  --tokenizer moses \
  --remove-bpe sentencepiece \
  --scoring sacrebleu \
  --batch-size 1 --langs $langs --beam 5  > "$MODEL_DIR"/test_b5.log


# You may want to comment out these lines. These lines exist because mBART checkpoints
# are massive (4.5 GB each), and two are generated at each savepoint.
printf "\n\nModels created by this run:"
ls $MODEL_DIR/*.pt
printf "\nDeleting these models for next run.\n"
rm $MODEL_DIR/*.pt
