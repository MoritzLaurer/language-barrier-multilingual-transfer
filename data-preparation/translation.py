

# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-b', '--batch_size', type=int,
                    help='batch_size for translations')
parser.add_argument('-ds', '--dataset', type=str,
                    help='Which dataset?')

args = parser.parse_args()

NMT_MODEL = args.nmt_model
BATCH_SIZE = args.batch_size
DATASET = args.dataset


## load packages
import pandas as pd
import numpy as np
from easynmt import EasyNMT
import tqdm  # https://github.com/tqdm/tqdm#documentation
import torch
import gc

def clean_memory():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()


#BATCH_SIZE = 64


## load df to translate
df_train = pd.read_csv(f"./data-clean/df_{DATASET}_train.csv", sep=",")   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',
df_test = pd.read_csv(f"./data-clean/df_{DATASET}_test.csv", sep=",")   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',

## drop duplicates
print(len(df_train))
print(len(df_test))
df_train = df_train[~df_train.text_original.duplicated(keep='first')]
df_test = df_test[~df_test.text_original.duplicated(keep='first')]
print(len(df_train))
print(len(df_test))

df_train = df_train.reset_index(drop=True)  # unnecessary nested index
df_test = df_test.reset_index(drop=True)  # unnecessary nested index


## take sample - do not need all data, since only using sample size of 1k (maybe 10k) to reduce excessive compute
# train
# at least 3 for each subcat to avoid algo issues downstream
df_train_samp_min_subcat = df_train.groupby(by="language_iso").apply(lambda x: x.groupby(by="label_subcat_text").apply(lambda x: x.sample(n=min(len(x), 3), random_state=42)))
df_train_samp = df_train.groupby(by="language_iso").apply(lambda x: x.sample(n=min(len(x), 10_000), random_state=42))
df_train_samp = pd.concat([df_train_samp, df_train_samp_min_subcat])
df_train_samp = df_train_samp[~df_train_samp.text_original.duplicated(keep='first')]
df_train_samp = df_train_samp.reset_index(drop=True)
print(len(df_train_samp))
print(df_train_samp.language_iso.value_counts())
# test
#df_test = df_test.groupby(by="language_iso").apply(lambda x: x.groupby(by="label_subcat_text").apply(lambda x: x.sample(n=min(len(x), 50), random_state=42)))
df_test_samp_min_subcat = df_test.groupby(by="language_iso").apply(lambda x: x.groupby(by="label_subcat_text").apply(lambda x: x.sample(n=min(len(x), 3), random_state=42)))
df_test_samp = df_test.groupby(by="language_iso").apply(lambda x: x.sample(n=min(len(x), 5_000), random_state=42))
df_test_samp = pd.concat([df_test_samp, df_test_samp_min_subcat])
df_test_samp = df_test_samp[~df_test_samp.text_original.duplicated(keep='first')]
df_test_samp = df_test_samp.reset_index(drop=True)
print(len(df_test_samp))
print(df_test_samp.language_iso.value_counts())


## translate each language in all other languages
# all parameters/methods for .translate here: https://github.com/UKPLab/EasyNMT/blob/main/easynmt/EasyNMT.py
#lang_lst_pimpo = ["sv", "no", "da", "fi", "nl", "es", "de", "en", "fr"]
lang_lst = ["en", "de", "es", "fr", "ko", "tr", "ru"]  #["eng", "deu", "spa", "fra", "kor", "jpn", "tur", "rus"]  #"ja"

# has to be M2M due to many language directions
model = EasyNMT(NMT_MODEL)  # m2m_100_418M,  m2m_100_1.2B, facebook/m2m100-12B-last-ckpt  opus-mt,

def translate_all2all(df=None, lang_lst=None, batch_size=8):
  df_step_lst = []
  for lang_target in tqdm.tqdm(lang_lst, desc="Overall all2all translation loop", leave=True):
    df_step = df[df.language_iso != lang_target].copy(deep=True)
    print("Translating texts from all other languages to: ", lang_target, ". ", len(df_step), " texts overall.")
    # specify source language to avoid errors. Automatic language detection can (falsely) identify languages that are not supported by model.
    for lang_source in tqdm.tqdm(np.sort(df_step.language_iso.unique()).tolist(), desc="Per source language loop", leave=True, position=2):
      df_step2 = df_step[df_step.language_iso == lang_source].copy(deep=True)
      print(f"    Translating {lang_source} to {lang_target}. {len(df_step2)} texts for this subset.")
      df_step2["text_original_trans"] = model.translate(df_step2["text_original"].tolist(), source_lang=lang_source, target_lang=lang_target, show_progress_bar=False, beam_size=5, batch_size=batch_size, perform_sentence_splitting=False)
      df_step2["language_iso_trans"] = [lang_target] * len(df_step2)
      df_step_lst.append(df_step2)
      clean_memory()
    #df_step["text_original_trans"] = model.translate(df_step["text_original"].tolist(), target_lang=lang_target, show_progress_bar=True, beam_size=5, batch_size=32, perform_sentence_splitting=False)
    #df_step_lst.append(df_step)
  return pd.concat(df_step_lst)


## translate test
df_test_samp_trans = translate_all2all(df=df_test_samp, lang_lst=lang_lst, batch_size=BATCH_SIZE)  # df[df.language.isin(["de", "en"])].sample(n=20, random_state=42)
# concatenate translated texts with original texts
print(len(df_test_samp_trans))
df_test_samp["text_original_trans"] = df_test_samp["text_original"]  #[np.nan] * len(df_test_samp)
df_test_samp["language_iso_trans"] = df_test_samp["language_iso"]  #[np.nan] * len(df_test_samp)
df_test_trans_concat = pd.concat([df_test_samp, df_test_samp_trans], axis=0)
df_test_trans_concat = df_test_trans_concat.drop_duplicates()
print(len(df_test_trans_concat))
# write to disk
df_test_trans_concat.to_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}.csv", index=False)

## translate test
df_train_samp_trans = translate_all2all(df=df_train_samp, lang_lst=lang_lst, batch_size=BATCH_SIZE)  # df[df.language.isin(["de", "en"])].sample(n=20, random_state=42)
# concatenate translated texts with original texts
print(len(df_train_samp_trans))
df_train_samp["text_original_trans"] = df_train_samp["text_original"]  #[np.nan] * len(df_train_samp)
df_train_samp["language_iso_trans"] = df_train_samp["language_iso"]  #[np.nan] * len(df_train_samp)
df_train_trans_concat = pd.concat([df_train_samp, df_train_samp_trans], axis=0)
df_train_trans_concat = df_train_trans_concat.drop_duplicates()
print(len(df_train_trans_concat))
# write to disk
df_train_trans_concat.to_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}.csv", index=False)




