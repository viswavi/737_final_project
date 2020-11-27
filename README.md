# Final project for 11-737
## (Better understanding monolingual data copying)
### by [@harlenesamra](https://github.com/harlenesamra), [@kinjaljain](https://github.com/kinjaljain), [@viswavi](https://github.com/viswavi)
TODO: rename the repo to have a more declarative name

## Table of contents (taken from [assignment 2 repo](https://github.com/viswavi/11-737-group/tree/master/assign2))
1. [General setup instructions](#general)
2. [Choosing new transfer languages](#harlene)
3. [Using phonetic intermediate representation for training](#kinjal_1)
4. [Training with the soft decoupled encoding model](#kinjal_2)
4. Performing data augmentation
    1. [Download monolingual data](#download_bt)
    2. [Preprocessing data with backtranslation or data copying](#preprocess_bt)
    3. [Training with augmented data](#train_bt)
    2. [Measuring repeated sequences in translations for post-hoc analysis](#repetitions)



## General Setup instructions (taken from project 2 handout)  <a name="general"></a>

### download data:
First, you can download the data using
```python
download_data.py
``` 

### fairseq:
The preprocess and experiment scripts work with fairseq. To use fairseq, first clone and install it from [here](https://github.com/pytorch/fairseq/).
You might need to install two more python packages. Simply do:
```
pip install importlib_metadata
```

```
pip install sacremoses
```

```
pip install sentencepiece
```

### preprocess data:
To preprocess the data for bilingual training, please do
```bash
preprocess_scripts/make-ted-bilingual.sh <aze/bel>
```

To preprocess the data for multilingual training, please do
```base
preprocess_scripts/make-ted-multilingual.sh <aze/bel> <tur/rus>
```

### training and translation scripts:
To submit training and translation experiment for bilingual setting for aze-eng, use the script
```bash
job_scripts/monolingual_trainer.sh <aze/bel> eng
```

To submit training and translation experiment for multilingual setting for aze-eng, use the script
```bash
job_scripts/multilingual_trainer.sh <aze/bel> <tur/rus> eng
```

To submit training and translation experiment for bilingual setting for eng-aze, use the script
```bash
job_scripts/monolingual_trainer.sh eng <aze/bel>
```

To submit training and translation experiment for multilingual setting for eng-aze, use the script
```bash
job_scripts/multilingual_trainer.sh eng <tur/rus> <aze/bel>
```
If you are using GPUs, remember to configure the CUDA_VISIBLE_DECIVE to different number if you are running muliple experiments at the same time.\

## Choosing new transfer languages   <a name="harlene"></a>
To preprocess the data for multilingual training, please do
```base
preprocess_scripts/make-ted-multilingual.sh <aze/bel> <ara/(ukr/pol)>
```
To submit training and translation experiment for multilingual setting for aze-eng, use the script
```bash
job_scripts/multilingual_trainer.sh aze ara eng
```
To submit training and translation experiment for multilingual setting for eng-aze, use the script
```bash
job_scripts/multilingual_trainer.sh eng ara aze
```
To submit training and translation experiment for multilingual setting for bel-eng, use the script
```bash
job_scripts/multilingual_trainer.sh bel <ukr/pol> eng
```
To submit training and translation experiment for multilingual setting for eng-bel, use the script
```bash
job_scripts/multilingual_trainer.sh bel <ukr/pol> eng
```

## Using phonetic intermediate representation for training  <a name="kinjal_1"></a>
To create data with IPA represeantation, run the script
```
python preprocess_scripts/convert_epitran.py data/ted_raw/aze_eng/ted-train.orig.aze-eng data/ted_raw/aze_eng/ted-train.orig.aze-eng

python preprocess_scripts/convert_epitran.py data/ted_raw/aze_eng/ted-dev.orig.aze-eng data/ted_raw/aze_eng/ted-dev.orig.aze-eng

python preprocess_scripts/convert_epitran.py data/ted_raw/aze_eng/ted-test.orig.aze-eng data/ted_raw/aze_eng/ted-test.orig.aze-eng
``` 
Run other preprocessing scripts as is before training the model.

## Training with the soft decoupled encoding model   <a name="kinjal_2"></a>
For this we used Xinyi's [codebase](https://github.com/cindyxinyiwang/SDE) with minor changes for preprocesssing data. We first generated data for aze-eng, tur-eng, bel-eng, rus-eng using make_data.sh and make-eng.sh sripts. Then, training and decoding was done as pointed out in the repository and BLEU score was calculated using this [script](https://github.com/pytorch/fairseq/fairseq_cli/score.py).

## How to use data augmentation via backtranslation or monolingual data copying to improve your models
There is evidence that using [backtranslation](https://arxiv.org/pdf/1808.09381.pdf) or [monolingual data copying](https://kheafield.com/papers/edinburgh/copy_paper.pdf) can improve low-resource NMT. In this section, we'll describe how to run either of these techniques with a few simple commands.

### Download monolingual data for backtranslation   <a name="download_bt"></a>
(I've committed the data into the repo, so can skip the rest of this section if desired, and jump to "Preprocessing...")

Download:\
https://wortschatz.uni-leipzig.de/en/download/azerbaijani (Newscrawl, 2013, 30K)\
https://wortschatz.uni-leipzig.de/en/download/belarusian/ (Newscrawl, 2017, 30K)\
https://wortschatz.uni-leipzig.de/en/download/english (Newscrawl-public, 2018, 30K)\
https://wortschatz.uni-leipzig.de/en/download/russian/ (Newscrawl-public, 2018, 30K)\
https://wortschatz.uni-leipzig.de/en/download/turkish (Newscrawl, 2018, 30K)\
https://wortschatz.uni-leipzig.de/en/download/marathi (Newscrawl 2016, 30K)\
https://wortschatz.uni-leipzig.de/en/download/kurdish (Newscrawl 2011, 30K)\
https://wortschatz.uni-leipzig.de/en/download/bengali (Newscrawl 2017, 30K)

```
mkdir data/monolingual_data
for lang in aze bel rus tur eng kur mar ben; do
    mkdir data/monolingual_data/${lang}
done
```
Then, untar each monolingual data file inside the corresponding monolingual data directory.

### Preprocessing with backtranslated or monolingually-augmented data   <a name="preprocess_bt"></a>
If you want to use backtranslation, run this script. If you simply want to do monolingual data augmentation, you may skip this step:
```
./preprocess_scripts/create_backtranslated_data_from_monolingual.sh <sampling/beam> < optional: num sentences to translate (6000 by default) >
```
This script requires that you have bilingual models in the fairseq/checkpoints directory for aze, bel, rus, and tur (both O2M and M2O). This script runs translation against monolingual data for all 8 language pairs, and takes at least an hour to generate translations for 6000 sentences per language pair.

Here, select whether you want to process data in "backtranslated" or "monoaugment" mode.
```
./preprocess_scripts/process_backtranslated_data.sh <backtranslated/monoaugment>
```
This script collects the output of the previous step and formats it into fairseq-friendly training data directories. We produce two separate data directories: one for O2M training files and one for M2O training (each marked with a suffix).

Then, process this augmented data for your desired training setup (bilingual or multilingual), and with the data augmentation option consistent with the above commands.

#### Biilingual models

```
./preprocess_scripts/make-ted-bilingual.sh <aze/bel> <backtranslated_for_M2O/backtranslated_for_O2M/monoaugment_for_M2O/monoaugment_for_O2M>
```

#### Multilingual models

```
preprocess_scripts/make-ted-multilingual.sh <aze/bel> <tur/rus> <backtranslated_for_M2O/backtranslated_for_O2M/monoaugment_for_M2O/monoaugment_for_O2M>
```

### Training with backtranslated or monolingually-augmented data   <a name="train_bt"></a>

#### Bilingual models
```
./job_scripts/monolingual_trainer.sh eng <aze/bel/rus/tur> <backtranslated_for_O2M/monoaugment_for_O2M>
```
or
```
./job_scripts/monolingual_trainer.sh <aze/bel/rus/tur> eng <backtranslated_for_M2O/monoaugment_for_M2O>
```

#### Multilingual models
```
./job_scripts/multilingual_trainer.sh eng <aze/bel> <tur/rus> <backtranslated_for_O2M/monoaugment_for_O2M>
```
or

```
./job_scripts/multilingual_trainer.sh <aze/bel> <tur/rus> eng <backtranslated_for_M2O/monoaugment_for_M2O>
```

### Measuring repeated sequences in translations   <a name="repetitions"></a>
In our analysis of backtranslation, we noticed that some of our worse translation models had the tendency to frequently generate a meaningless sequence of characters. In an effort to measure the frequency that this happens, we wrote a script to help::
```
./preprocess_scripts/find_repeated_substrings.py <path_to_translation_output_file>
```

Simply pass in a file containing the stdout produced by `fairseq-generate` or `fairseq-interactive`, and this script will print out the percentage of translated sentences containing a sequence of characters repeated N times (where N ranges from 5 to 30, in increments of 5).
