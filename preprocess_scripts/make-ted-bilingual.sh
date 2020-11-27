#!/bin/bash

if [ ! -d mosesdecoder ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

if [ $# -ge 1 ]; then
    srclang=$1
    if [ "$srclang" = "eng" ]; then
        echo "You only need to run this once from your source language to 
        target language - don't need to run it both ways!"
        exit 1
    fi
    # Can supply a suffix (e.g. if we want to create separate data directories for data
    # produced via data augmentation)
    if [ $# -eq 2 ]; then
      suffix=_$2
    else
      suffix=
    fi
else
    echo 'Error: Must provide a source language code (e.g. aze or bel)
     to prepare data for translating to English'
    exit 1
fi

VOCAB_SIZE=8000
RAW_DDIR=data${suffix}/ted_raw/
PROC_DDIR=data${suffix}/ted_processed/${srclang}_spm"$VOCAB_SIZE"/
BINARIZED_DDIR=fairseq/data-bin${suffix}/ted_${srclang}_spm"$VOCAB_SIZE"/
FAIR_SCRIPTS=fairseq/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
TOKENIZER=mosesdecoder/scripts/tokenizer/tokenizer.perl

LANS=(
  $srclang)

for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}
  mkdir -p "$PROC_DDIR"/"$LAN"_eng
  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.*-eng  ; do
    src=`echo $f | sed 's/-eng$//g'`
    trg=`echo $f | sed 's/\.[^\.]*$/.eng/g'`

    if [ ! -f "$src" ]; then
      echo "src=$src, trg=$trg"
      python preprocess_scripts/cut-corpus.py 0 < $f > $src
      python preprocess_scripts/cut-corpus.py 1 < $f > $trg
    fi
  done
  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.{eng,$LAN} ; do
    f1=${f/orig/mtok}
    if [ ! -f "$f1" ]; then
      echo "tokenize $f1..."
      cat $f | perl $TOKENIZER > $f1
    fi
  done
  # learn BPE with sentencepiece
  TRAIN_FILES="$RAW_DDIR"/"$LAN"_eng/ted-train.mtok."$LAN","$RAW_DDIR"/"$LAN"_eng/ted-train.mtok.eng
  echo "learning joint BPE over ${TRAIN_FILES}..."
  python "$SPM_TRAIN" \
	--input=$TRAIN_FILES \
	--model_prefix="$PROC_DDIR"/"$LAN"_eng/spm"$VOCAB_SIZE" \
	--vocab_size=$VOCAB_SIZE \
	--character_coverage=1.0 \
	--model_type=bpe

  python "$SPM_ENCODE" \
	--model="$PROC_DDIR"/"$LAN"_eng/spm"$VOCAB_SIZE".model \
	--output_format=piece \
	--inputs "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok."$LAN" "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok.eng  \
	--outputs "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE"."$LAN" "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".eng \
	--min-len 1 --max-len 200 
 
  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
  python "$SPM_ENCODE" \
	--model="$PROC_DDIR"/"$LAN"_eng/spm"$VOCAB_SIZE".model \
	--output_format=piece \
	--inputs "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok."$LAN" "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok.eng  \
	--outputs "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE"."$LAN" "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE".eng  
  done

  echo "Binarize the data..."
  fairseq-preprocess --source-lang $LAN --target-lang eng \
	--joined-dictionary \
	--trainpref "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE" \
	--validpref "$PROC_DDIR"/"$LAN"_eng/ted-dev.spm"$VOCAB_SIZE" \
	--testpref "$PROC_DDIR"/"$LAN"_eng/ted-test.spm"$VOCAB_SIZE" \
	--destdir $BINARIZED_DDIR/"$LAN"_eng/

  echo "Binarize the data..."
  fairseq-preprocess --source-lang eng --target-lang $LAN \
	--joined-dictionary \
	--trainpref "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE" \
	--validpref "$PROC_DDIR"/"$LAN"_eng/ted-dev.spm"$VOCAB_SIZE" \
	--testpref "$PROC_DDIR"/"$LAN"_eng/ted-test.spm"$VOCAB_SIZE" \
	--destdir $BINARIZED_DDIR/eng_"$LAN"/

done
