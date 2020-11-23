#!/bin/bash

if [ $# -ge 3 ]; then
    srclang=$1
    hrlang=$2
    trglang=$3
	if [ "$trglang" = "eng" ]; then
		lrlang=$srclang
		# M2O means translating languages to English ("many 2 one")
		multi_type="M2O"
		encoder_langtok="src"
		lang_pairs=${srclang}-${trglang},${hrlang}-${trglang}
	else
		lrlang=$trglang
		# O2M means translating languages from English ("one 2 many")
		multi_type="O2M"
		encoder_langtok="tgt"
		lang_pairs=${srclang}-${trglang},${srclang}-${hrlang}
	fi
	if [ $# -eq 4 ]; then
	    model_directory_suffix=_$4
	else
	    model_directory_suffix=
	fi
else
    echo 'Error: Must provide a source language code (e.g. aze or bel), a 
	high-resource language (e.g. tur or rus) to perform multilingual training 
	with, and a target language (e.g. eng) to translate to.'
    exit 1
fi

DATA_DIR=fairseq/data-bin${model_directory_suffix}/ted_${lrlang}${hrlang}_sepspm8000/${multi_type}/
MODEL_DIR=fairseq/checkpoints${model_directory_suffix}/ted_${lrlang}${hrlang}_sepspm8000/${multi_type}/
mkdir -p $MODEL_DIR
GPU_DEVICE=0

# train the model
CUDA_VISIBLE_DEVICE=$GPU_DEVICE fairseq-train \
	$DATA_DIR \
	--arch transformer_iwslt_de_en \
	--task translation_multi_simple_epoch \
	--encoder-langtok ${encoder_langtok} \
	--lang-pairs $lang_pairs \
	--max-epoch 40 \
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
CUDA_VISIBLE_DEVICE=$GPU_DEVICE fairseq-generate $DATA_DIR \
    --gen-subset test \
	--task translation_multi_simple_epoch \
	--lang-pairs $lang_pairs \
	--encoder-langtok ${encoder_langtok} \
	--source-lang $srclang \
	--target-lang $trglang \
	--path $MODEL_DIR/checkpoint_best.pt \
	--batch-size 32 \
	--tokenizer moses \
	--remove-bpe sentencepiece \
	--scoring sacrebleu \
	--beam 5   > "$MODEL_DIR"/test_b5.log


CUDA_VISIBLE_DEVICE=$GPU_DEVICE fairseq-generate $DATA_DIR \
	--gen-subset valid \
	--task translation_multi_simple_epoch \
	--lang-pairs $lang_pairs \
	--encoder-langtok ${encoder_langtok} \
	--source-lang $srclang --target-lang $trglang \
	--path $MODEL_DIR/checkpoint_best.pt \
	--batch-size 32 \
	--tokenizer moses \
	--remove-bpe sentencepiece \
	--scoring sacrebleu \
	--beam 5   > "$MODEL_DIR"/valid_b5.log


