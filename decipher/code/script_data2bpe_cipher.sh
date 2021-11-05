#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

set -e

SCRIPTS=code/tools/mosesdecoder/scripts

if [ ! -d "$SCRIPTS" ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  cd code/tools/
  git clone https://github.com/moses-smt/mosesdecoder.git
  cd ../../
fi

TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPE_TOKENS=40000

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

LANG0_LANG1=(
  "en ci" # ci stands for 'cipher'
  "ci en" # ci stands for 'cipher'
)

root_in=data/cipher/data/cau2eff
root_out=data/cipher/data/cau2eff_for_nmt
mkdir $root_out || true

for lang_pair in "${LANG0_LANG1[@]}"; do
  set -- $lang_pair
  lang0=$1
  lang1=$2

  # example_file=${root}/conf100_en-de/train_unsup.de
  raw_files=(
    "${root_in}/$lang0-$lang1.$lang0"
    "${root_in}/$lang0-$lang1.$lang1"
  )

  # Step 0. Copy files to the new folder ${folder_out}
  i=0
  for f in "${raw_files[@]}"; do
    new_f="${f/./\/}"
    new_f="${new_f/cau2eff/cau2eff_for_nmt/}"
    folder_out=$(dirname "$new_f")
    mkdir -p "$folder_out"
    cp $f $new_f
    FILES[i++]="$new_f"
  done

  echo "Preparing ${FILES[@]}"

  # Step 1. Clean punctuations & tokenize
  for f in ${FILES[@]}; do
    f_lang=${f: -2} # get the language code of file f
    f_lang='en'
    cat $f |
      perl $NORM_PUNC ${f_lang} |
      perl $REM_NON_PRINT_CHAR |
      perl $TOKENIZER -threads 8 -a -l ${f_lang} >${f}.tok
  done

  # Step 2. Separate out {valid, test}
  echo "splitting test, valid, train, unsup by 2K:2K:96K:900K ..."
  for f in ${FILES[@]}; do
    awk '{if (NR%100 == 0)  print $0; }' $f.tok >$f.tok.test
    awk '{if (NR%100 == 1)  print $0; }' $f.tok >$f.tok.valid
    awk '{if (NR%100 != 0 && NR%100 != 1 && NR%100 <= 50)  print $0; }' $f.tok >$f.tok.train
    awk '{if (NR%100 > 50)  print $0; }' $f.tok >$f.tok.unsup
  done

  # Update the file list from raw to ".tok"
  i=0
  j=0
  for f in ${FILES[@]}; do
    FILES_TOK_TEST[j++]=$f.tok.test
    FILES_TOK_TEST[j++]=$f.tok.valid
    FILES_TOK_NONTEST[i++]=$f.tok.train
    FILES_TOK_NONTEST[i++]=$f.tok.unsup
  done
  # echo "${FILES_TOK_NONTEST[@]}"
  # echo "-----"
  # echo "${FILES_TOK_TEST[@]}"

  # Step 3. Learn BPE and apply BPE
  for lang in $lang0 $lang1; do
    bpe_corp=${folder_out}/${lang}.bpe_corp
    BPE_CODE=${folder_out}/${lang}.bpe_code
    echo >$bpe_corp

    f_lang_tmpl=$lang.tok
    echo $f_lang_tmpl

    for f in ${FILES_TOK_NONTEST[@]}; do
      if [[ $f =~ $f_lang_tmpl ]]; then
        echo "cat $f >>$bpe_corp"
        cat $f >>$bpe_corp
      fi

    done

    echo "learn_bpe.py on ${bpe_corp} which has the following number of lines:"
    wc -l $bpe_corp

    subword-nmt learn-bpe -s $BPE_TOKENS <$bpe_corp >$BPE_CODE

    TOK_FILES=("${FILES_TOK_NONTEST[@]}" "${FILES_TOK_TEST[@]}")
    for f in ${TOK_FILES[@]}; do
      if [[ $f =~ $f_lang_tmpl ]]; then
        echo "apply_bpe.py to ${f} ..."
        subword-nmt apply-bpe -c $BPE_CODE <$f >$f.bpe
      fi
    done
  done

  # Step 4. Clean by 1.5x length ratio
  i=0
  for f in "${FILES_TOK_NONTEST[@]}"; do
    f=$f.bpe
    dir=$(dirname $f)

    f_name=$(basename $f)
    f_lang=${f_name:0:2} # len('en') == 2
    f_split=${f_name:3}

    new_f_name=tmp.$f_split.$f_lang
    new_f=$dir/$new_f_name
    FILES_BPE_NONTEST[i++]="$new_f"

    cp $f $new_f
  done

  for f in ${FILES_BPE_NONTEST[@]}; do
    f_name=$(basename $f)
    dir=$(dirname $f)

    f_name_len=${#f_name}
    f_split_len=($f_name_len-4-3)
    f_split=${f_name:4:$f_split_len} # len('tmp.') == 4, len('.en') == 3
    f_lang=${f_name: -2}
    ori_f=$f_lang.$f_split

    ori_f=$dir/$ori_f
    f_out=$ori_f.fitlen

    f_split_len=($f_split_len-4-4)
    f_out_rename=${f_split:4:$f_split_len} # len('tok.test.bpe') == $f_split_len

    f_in=$dir/${f_name:0:($f_name_len - 3)}

#    perl $CLEAN -ratio 1.5 ${f_in} $lang0 $lang1 $f_in.fitlen 1 250

#    for lang in $lang0 $lang1; do
#      mv $f_in.$lang $dir/$lang.$f_split
#      cp $dir/$lang.$f_split $dir/$f_out_rename.$lang
#    done

#    echo "Output file with <=1.5 length ratio: $dir/$f_out_rename.$lang0 $dir/$f_out_rename.$lang1 "
  done

  for split in "train" "unsup" "valid" "test"; do
    cp $dir/$lang0.tok.$split.bpe $dir/$split.$lang0
    cp $dir/$lang1.tok.$split.bpe $dir/$split.$lang1
  done
#  split='train'
#  for lang in $lang0 $lang1; do
#    head -30000 $dir/$split.$lang >$dir/${split}_30k.$lang
#    head -50000 $dir/$split.$lang >$dir/${split}_50k.$lang
#  done
#  split='unsup'
#  for lang in $lang0 $lang1; do
#    head -30000 $dir/$split.$lang >$dir/${split}_30k.$lang
#    head -50000 $dir/$split.$lang >$dir/${split}_50k.$lang
#    head -500000 $dir/$split.$lang >$dir/${split}_500k.$lang
#  done

  for f in ${FILES_BPE_NONTEST[@]}; do
    rm $f
  done

done
