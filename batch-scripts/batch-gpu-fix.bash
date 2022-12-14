#!/bin/bash
# Set batch job requirements
#SBATCH -t 20:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=gpu-fix

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
cd ./multilingual-repo

# install packages
pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# for local run
#bash ./batch-scripts/batch-gpu.bash

## scenarios
# "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
# "tfidf", "embeddings-en", "embeddings-multi"

study_date=221101
sample=500
#n_trials=10
#n_trials_sampling=7
#n_trials_pruning=7
#n_cross_val_hyperparam=2
n_cross_val_final=3
model='transformer'
method='standard_dl'
dataset='manifesto-8'
nmt_model='m2m_100_418M'  # m2m_100_418M, m2m_100_1.2B



## scenario loops
# these two scenarios only works for embeddings-multi
translation_lst='no-nmt-single one2many'
vectorizer_lst='embeddings-multi'

for translation in $translation_lst
do
  for vectorizer in $vectorizer_lst
  do
    python analysis-transf-run.py --n_cross_val_final $n_cross_val_final \
           --dataset $dataset --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" --nmt_model $nmt_model \
           --augmentation_nmt $translation --model $model --vectorizer $vectorizer --method $method \
           --sample_interval $sample --hyperparam_study_date $study_date  &> ./results/manifesto-8/logs/run-$model-$translation-$vectorizer-$sample-$dataset-$nmt_model-$study_date-logs-fix.txt
    echo Final run done for scenario: $translation $vectorizer
  done
done

## scenario loop
#translation_lst='one2anchor no-nmt-many many2anchor many2many'
#vectorizer_lst='embeddings-en embeddings-multi'
translation_lst='many2many'
vectorizer_lst='embeddings-multi'

echo Starting training loop
for translation in $translation_lst
do
  for vectorizer in $vectorizer_lst
  do
    python analysis-transf-run.py --n_cross_val_final $n_cross_val_final \
           --dataset $dataset --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" --nmt_model $nmt_model \
           --augmentation_nmt $translation --model $model --vectorizer $vectorizer --method $method \
           --sample_interval $sample --hyperparam_study_date $study_date  &> ./results/manifesto-8/logs/run-$model-$translation-$vectorizer-$sample-$dataset-$nmt_model-$study_date-logs-fix.txt
    echo Final run done for scenario: $translation $vectorizer
  done
done



echo Entire script done

