#!/bin/bash

if [ $# -ge 2 ]; then
    srclang=$1
    trglang=$2
	if [ "$trglang" = "eng" ]; then
		lrlang=$srclang
	else
		lrlang=$trglang
	fi

	if [ $# -ge 3 ]; then
	    model_directory_suffix=_$3
	else
	    model_directory_suffix=
	fi

	if [ $# -eq 4 ]; then
	    num_epochs=$4
	else
	    num_epochs=80
	fi
else
    echo 'Error: Must provide a source language code (e.g. aze or bel) 
    and a target language code (e.g. eng)'
    exit 1
fi

DATA_DIR=fairseq/data-bin${model_directory_suffix}/ted_${lrlang}_spm8000/${srclang}_${trglang}/
MODEL_DIR=fairseq/checkpoints${model_directory_suffix}/ted_${lrlang}_spm8000/${srclang}_${trglang}/
mkdir -p $MODEL_DIR
GPU_DEVICE=0

# change the cuda_visible_device to the GPU device number you are using
# train the model
CUDA_VISIBLE_DEVICE=$GPU_DEVICE fairseq-train \
	$DATA_DIR \
	--arch transformer_iwslt_de_en \
	--max-epoch $num_epochs \
        --distributed-world-size 1 \
	--share-all-embeddings \
	--no-epoch-checkpoints \
	--dropout 0.3 \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	--criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	--max-tokens 4500 \
	--update-freq 2 \
	--seed 2 \
	--save-dir $MODEL_DIR \
	--log-interval 100 >> $MODEL_DIR/train.log 2>&1

# translate the valid and test set
CUDA_VISIBLE_DEVICE=$GPU_DEVICE  fairseq-generate $DATA_DIR \
          --gen-subset test \
          --path $MODEL_DIR/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/test_b5.log


CUDA_VISIBLE_DEVICE=$GPU_DEVICE fairseq-generate $DATA_DIR \
          --gen-subset valid \
          --path $MODEL_DIR/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/valid_b5.log


