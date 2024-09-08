#camille test
#new camille test - hello
import sys
if len(sys.argv) > 1:
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (len(sys.argv): ", len(sys.argv), ")")


### Load packages
import transformers
import datasets
import torch
import optuna
import pandas as pd
import numpy as np
import re
import math
import random
import os
import tqdm
from collections import OrderedDict
import joblib
from datetime import date
from datetime import datetime
import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import svm, naive_bayes, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import spacy


## set global seed for reproducibility and against seed hacking
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

print(os.getcwd())
if (EXECUTION_TERMINAL==False) and ("multilingual-repo" not in os.getcwd()):
    os.chdir("./multilingual-repo")
print(os.getcwd())


# ## Main parameters

### argparse for command line execution
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run hyperparameter tuning with different languages, algorithms, datasets')

## Add the arguments
# arguments for hyperparameter search
parser.add_argument('-t', '--n_trials', type=int,
                    help='How many optuna trials should be run?')
parser.add_argument('-ts', '--n_trials_sampling', type=int,
                    help='After how many trials should optuna start sampling?')
parser.add_argument('-tp', '--n_trials_pruning', type=int,
                    help='After how many trials should optuna start pruning?')
parser.add_argument('-cvh', '--n_cross_val_hyperparam', type=int, default=2,
                    help='How many times should optuna cross validate in a single trial?')

# arguments for both hyperparam and test script
#parser.add_argument('-lang', '--languages', type=str, nargs='+',
#                    help='List of languages to iterate over. e.g. "en", "de", "es", "fr", "ko", "tr", "ru" ')
#parser.add_argument('-anchor', '--language_anchor', type=str,
#                    help='Anchor language to translate all texts to if not many2many. Default is "en"')
#parser.add_argument('-language_train', '--language_train', type=str, nargs='+',
parser.add_argument('-languages', '--languages', type=str,  #nargs='+',
                    help='What language should the training set be in?. Default is "en"')
parser.add_argument('-augment', '--augmentation_nmt', type=str,
                    help='Whether and how to augment the data with machine translation (MT).')

parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "manifesto-8" ')
parser.add_argument('-samp', '--max_sample_lang', type=int, nargs='+',
                    help='Interval of sample sizes to test.')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "classical_ml"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings"')
parser.add_argument('-hp_date', '--study_date', type=str,
                    help='Date string to specifiy which hyperparameter run should be selected. e.g. "20220304"')

parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-ta', '--task', type=str,
                    help='task about integration or immigration?')


### choose arguments depending on execution in terminal or in script for testing
if EXECUTION_TERMINAL == True:
  print("Arguments passed via the terminal:")
  # Execute the parse_args() method
  args = parser.parse_args()
  # To show the results of the given option to screen.
  print("")
  for key, value in parser.parse_args()._get_kwargs():
      #if value is not None:
          print(value, "  ", key)

elif EXECUTION_TERMINAL == False:
  # parse args if not in terminal, but in script
  args = parser.parse_args(["--n_trials", "3", "--n_trials_sampling", "2", "--n_trials_pruning", "2", "--n_cross_val_hyperparam", "2",  #"--context",
                            #"--dataset", "pimpo",
                            #"--languages", 'sv', 'no', 'da', 'fi', 'nl', 'es', 'de', 'en', 'fr', #  "en", "de", "es", "fr", "tr", "ru", "ko",  # 'sv' 'no' 'da' 'fi' 'nl' 'es' 'de' 'en' 'fr'
                            #"--language_anchor", "en",
                            "--languages", "en", "--task", "immigration",
                            #"--augmentation_nmt", "one2anchor",  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
                            "--max_sample_lang", "100",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "dl_embed", #"--model", "logistic",  # SVM, logistic
                            "--vectorizer", "multi",  # "tfidf", "en", "multi"
                            "--nmt_model", "m2m_100_1.2B",  # "m2m_100_1.2B", "m2m_100_418M"
                            "--study_date", "221111"])


### args only for hyperparameter tuning
N_TRIALS = args.n_trials
N_STARTUP_TRIALS_SAMPLING = args.n_trials_sampling
N_STARTUP_TRIALS_PRUNING = args.n_trials_pruning
CROSS_VALIDATION_REPETITIONS_HYPERPARAM = args.n_cross_val_hyperparam
#CONTEXT = False  #args.context   # ! do not use context, because some languages have too little data and makes train-test split problematic

### args for both hyperparameter tuning and test runs
# choose dataset
#DATASET = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
DATASET = "pimpo"
TASK = args.task
#LANGUAGES = args.languages
#LANGUAGES = LANGUAGES
LANGUAGE_LST = args.languages.split("-")
#LANGUAGE_TRAIN = args.languages
#LANGUAGE_TRAIN = LANGUAGE_TRAIN.split("-")
#LANGUAGE_ANCHOR = args.language_anchor
#AUGMENTATION = args.augmentation_nmt

N_SAMPLE_DEV = args.max_sample_lang   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "dl_embed"
MODEL_NAME = "logistic"  #args.model  # "SVM"

HYPERPARAM_STUDY_DATE = args.study_date  #"20220304"
NMT_MODEL = args.nmt_model





# ## Load data
if "pimpo" in DATASET:
  df_cl = pd.read_csv("./data-clean/df_pimpo_all.zip", engine='python')
  df_train = pd.read_csv(f"./data-clean/df_{DATASET}_samp_trans_{NMT_MODEL}_embed_tfidf.zip", engine='python')
  # only doing analysis on immigration
  if TASK == "immigration":
    df_cl = df_cl[df_cl.label_text.isin(['no_topic', 'immigration_sceptical', 'immigration_supportive', 'immigration_neutral'])]  # 'integration_neutral', 'integration_sceptical', 'integration_supportive'
    df_train = df_train[df_train.label_text.isin(['no_topic', 'immigration_sceptical', 'immigration_supportive', 'immigration_neutral'])]  # 'integration_neutral', 'integration_sceptical', 'integration_supportive'
  else:
      raise Exception(f"Task {TASK} not implemented.")
else:
  raise Exception(f"Dataset name not found: {DATASET}")

print(DATASET)




## reduce max sample size interval list to fit to max df_train length
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len(df_train)]
if len(df_train) < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len(df_train)]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)


LABEL_TEXT_ALPHABETICAL = np.sort(df_cl.label_text.unique())
TRAINING_DIRECTORY = f"results/{DATASET}"




# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())

import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)

from helpers import compute_metrics_classical_ml, clean_memory

## functions for scenario data selection and augmentation
#from helpers import select_data_for_scenario_hp_search, sample_for_scenario_hp, choose_preprocessed_text






##### hyperparameter tuning

# carbon tracker  https://github.com/mlco2/codecarbon/tree/master
#if CARBON_TRACKING:
#  from codecarbon import OfflineEmissionsTracker
#  tracker = OfflineEmissionsTracker(country_iso_code="NLD",  log_level='warning', measure_power_secs=300,  #output_dir=TRAINING_DIRECTORY
#                                    project_name=f"{DATASET}-{MODEL_NAME.split('/')[-1]}")
#  tracker.start()


def optuna_objective(trial, lang=None, n_sample=None, df_train=None, df=None):  # hypothesis_hyperparams_dic=None
  clean_memory()
  np.random.seed(SEED_GLOBAL)  # setting seed again for safety. not sure why this needs to be run here at each iteration. it should stay constant once set globally?! explanation could be this https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f

  # for testing / debugging
  global df_train_split
  global df_train_scenario
  global df_train_scenario_samp
  global df_train_scenario_samp_augment
  global df_dev_scenario
  global df_dev_scenario_samp
  #global df_train_scenario_samp_ids
  #global df_dev_scenario_samp_ids

  if VECTORIZER == "tfidf":
      # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
      hyperparams_vectorizer = {
          'ngram_range': trial.suggest_categorical("ngram_range", [(1, 2), (1, 3), (1, 6)]),
          'max_df': trial.suggest_categorical("max_df", [1.0, 0.9, 0.8]),  # new runs for 'no-nmt-many many2many' + tfidf - others [0.95, 0.9, 0.8]  # can lead to error "ValueError: After pruning, no terms remain. Try a lower min_df or a higher max_df."
          'min_df': trial.suggest_categorical("min_df", [1, 0.01, 0.03]),  # new runs for 'no-nmt-many many2many' + tfidf - others [0.01, 0.03, 0.05]
          'analyzer': trial.suggest_categorical("analyzer", ["word", "char_wb"]),  # could be good for languages like Korean where longer sequences of characters without a space seem to represent compound words
      }
      vectorizer_sklearn = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"
  elif VECTORIZER in ["en", "multi"]:
      vectorizer_sklearn = ["somerandomstringtopreventerrorsfromoptuna"]
      hyperparams_vectorizer = {}

  # SVM  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  if MODEL_NAME == "SVM":
      hyperparams_clf = {'kernel': trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                   'C': trial.suggest_float("C", 1, 1000, log=True),
                   "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                   "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                   "coef0": trial.suggest_float("coef0", 1, 100, log=True),  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
                   "degree": trial.suggest_int("degree", 1, 50, log=False),  # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
                   #"decision_function_shape": trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"]),  # "However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary classification."
                   #"tol": trial.suggest_categorical("tol", [1e-3, 1e-4]),
                   "random_state": SEED_GLOBAL,
                    # 10k max_iter had 1-10% performance drop for high n sample (possibly overfitting to majority class)
                    #MAX_ITER_LOW, MAX_ITER_HIGH = 1_000, 7_000  # tried 10k, but led to worse performance on larger, imbalanced dataset (?)
                    "max_iter": trial.suggest_int("num_train_epochs", 1_000, 7_000, log=False, step=1000),  #MAX_ITER,
                   }
  # Logistic Regression # ! disadvantage: several parameters only work with certain other parameters  # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn-linear-model-logisticregression
  elif MODEL_NAME == "logistic":
      hyperparams_clf = {#'penalty': trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"]),
                        'penalty': 'l2',  # works with all solvers
                        'solver': trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
                        'C': trial.suggest_float("C", 1, 1000, log=False),
                        #"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                        #"intercept_scaling": trial.suggest_float("intercept_scaling", 1, 50, log=False),
                        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                        "max_iter": trial.suggest_int("max_iter", 50, 1000, log=False),  # 100 default
                        "multi_class": "auto",  # {‘auto’, ‘ovr’, ‘multinomial’}
                        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
                        #"l1_ratio": None,
                        "n_jobs": -1,
                        "random_state": SEED_GLOBAL,
                        }
  else:
      raise Exception("Method not available: ", MODEL_NAME)


  hyperparams_optuna = {**hyperparams_clf, **hyperparams_vectorizer}  # "hypothesis_template": hypothesis_template
  trial.set_user_attr("hyperparameters_all", hyperparams_optuna)
  print("Hyperparameters for this run: ", hyperparams_optuna)


  # cross-validation loop. Objective: determine F1 for specific sample for specific hyperparams, without a test set
  run_info_dic_lst = []
  for step_i, random_seed_cross_val in enumerate(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_HYPERPARAM)):

    ## train-validation split
    # ~50% split cross-val as recommended by https://arxiv.org/pdf/2109.12742.pdf
    test_size = 0.4

    df_train['stratify_by'] = df_train['language_iso'].astype(str) + "_" + df_train['label_text'].astype(str)
    df_train_split, df_dev = train_test_split(df_train, test_size=test_size, shuffle=True, random_state=random_seed_cross_val, stratify=df_train['stratify_by'])
    #df_train_split = df_train[df_train.rn.isin(df_train_split_ids)]
    #df_dev = df_train[df_train.rn.isin(df_dev_ids)]
    print(f"Final train test length after cross-val split: len(df_train_lang_samp) = {len(df_train_split)}, len(df_dev_lang_samp) {len(df_dev)}.")

    #df_train_scenario, df_dev_scenario = select_data_for_scenario_hp_search(df_train=df_train_split, df_dev=df_dev, lang=lang, augmentation=AUGMENTATION, vectorizer=VECTORIZER, language_train=LANGUAGE_LST, language_anchor=LANGUAGE_ANCHOR)
    # no augmentation here, so can just select by training languages
    df_train_scenario = df_train_split[df_train_split.language_iso.isin(LANGUAGE_LST)]
    df_dev_scenario = df_dev[df_dev.language_iso.isin(LANGUAGE_LST)]

    n_no_topic_or_topic = int(n_sample / 2)
    df_train_scenario_samp = df_train_scenario.groupby(by="language_iso", as_index=False, group_keys=False).apply(
        lambda x: pd.concat([x[x.label_text == "no_topic"].sample(n=min(int(n_no_topic_or_topic*(1-test_size)), len(x[x.label_text != "no_topic"])), random_state=random_seed_cross_val),
                             x[x.label_text != "no_topic"].sample(n=min(int(n_no_topic_or_topic*(1-test_size)), len(x[x.label_text != "no_topic"])), random_state=random_seed_cross_val)
                             ]))
    df_dev_scenario_samp = df_dev_scenario.groupby(by="language_iso", as_index=False, group_keys=False).apply(
        lambda x: pd.concat([x[x.label_text == "no_topic"].sample(n=min(int(n_no_topic_or_topic*(test_size)), len(x[x.label_text != "no_topic"])), random_state=random_seed_cross_val),
                             x[x.label_text != "no_topic"].sample(n=min(int(n_no_topic_or_topic*(test_size)), len(x[x.label_text != "no_topic"])), random_state=random_seed_cross_val)
                             ]))


    print("Number of training examples after sampling: ", len(df_train_scenario_samp))
    print("Label distribution for df_train_scenario_samp: ", df_train_scenario_samp.label_text.value_counts())
    print("Number of validation examples after sampling: ", len(df_dev_scenario_samp))
    print("Label distribution for df_dev_scenario_samp: ", df_dev_scenario_samp.label_text.value_counts())
    print("\n")

    clean_memory()
    ## choose correct pre-processed text column here
    # possible vectorizers: "tfidf", "embeddings-en", "embeddings-multi"
    #X_train, X_test = choose_preprocessed_text(df_train_scenario_samp_augment=df_train_scenario_samp, df_test_scenario=df_dev_scenario_samp, augmentation=AUGMENTATION, vectorizer=VECTORIZER, vectorizer_sklearn=vectorizer_sklearn, language_train=LANGUAGE_LST, language_anchor=LANGUAGE_ANCHOR, method=METHOD)
    if VECTORIZER == "tfidf":
        # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
        vectorizer_sklearn.fit(pd.concat([df_train_scenario_samp.text_trans_concat_tfidf, df_dev_scenario_samp.text_trans_concat_tfidf]))
        X_train = vectorizer_sklearn.transform(df_train_scenario_samp.text_trans_concat_tfidf)
        X_test = vectorizer_sklearn.transform(df_dev_scenario_samp.text_trans_concat_tfidf)
    elif "en" == VECTORIZER:
        X_train = np.array([ast.literal_eval(lst) for lst in df_train_scenario_samp.text_trans_concat_embed_en.astype('object')])
        X_test = np.array([ast.literal_eval(lst) for lst in df_dev_scenario_samp.text_trans_concat_embed_en.astype('object')])
    elif "multi" == VECTORIZER:
        X_train = np.array([ast.literal_eval(lst) for lst in df_train_scenario_samp.text_concat_embed_multi.astype('object')])
        X_test = np.array([ast.literal_eval(lst) for lst in df_dev_scenario_samp.text_concat_embed_multi.astype('object')])


    y_train = df_train_scenario_samp.label
    y_test = df_dev_scenario_samp.label

    # training on train set sample
    if MODEL_NAME == "SVM":
        clf = svm.SVC(**hyperparams_clf)
    elif MODEL_NAME == "logistic":
        clf = linear_model.LogisticRegression(**hyperparams_clf)
    clf.fit(X_train, y_train)

    # prediction on test set
    label_gold = y_test
    label_pred = clf.predict(X_test)
    
    # metrics
    metric_step = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df.label_text.unique()))

    run_info_dic = {"method": METHOD, "vectorizer": VECTORIZER, "augmentation": None, "n_sample": n_sample, "model": MODEL_NAME, "results": metric_step, "hyper_params": hyperparams_optuna}
    run_info_dic_lst.append(run_info_dic)
    
    # Report intermediate objective value.
    intermediate_value = (metric_step["eval_f1_macro"] + metric_step["eval_f1_micro"]) / 2
    trial.report(intermediate_value, step_i)
    # Handle trial pruning based on the intermediate value.
    if trial.should_prune() and (CROSS_VALIDATION_REPETITIONS_HYPERPARAM > 1):
      raise optuna.TrialPruned()
    if n_sample == 999_999:  # no cross-validation necessary for full dataset
      break


  ## aggregation over cross-val loop
  f1_macro_crossval_lst = [dic["results"]["eval_f1_macro"] for dic in run_info_dic_lst]
  f1_micro_crossval_lst = [dic["results"]["eval_f1_micro"] for dic in run_info_dic_lst]
  accuracy_balanced_crossval_lst = [dic["results"]["eval_accuracy_balanced"] for dic in run_info_dic_lst]
  metric_details = {
      "F1_macro_mean": np.mean(f1_macro_crossval_lst), "F1_micro_mean": np.mean(f1_micro_crossval_lst), "accuracy_balanced_mean": np.mean(accuracy_balanced_crossval_lst),
      "F1_macro_std": np.std(f1_macro_crossval_lst), "F1_micro_std": np.std(f1_micro_crossval_lst), "accuracy_balanced_std": np.std(accuracy_balanced_crossval_lst)
  }
  trial.set_user_attr("metric_details", metric_details)

  results_lst = [dic["results"] for dic in run_info_dic_lst]
  trial.set_user_attr("results_trainer", results_lst)

  # objective: maximise mean of f1-macro & f1-micro. HP should be good for imbalanced data, but also important/big classes
  metric = (np.mean(f1_macro_crossval_lst) + np.mean(f1_micro_crossval_lst)) / 2
  std = (np.std(f1_macro_crossval_lst) + np.std(f1_micro_crossval_lst)) / 2

  print(f"\nFinal metrics for run: {metric_details}. With hyperparameters: {hyperparams_optuna}\n")

  return metric



def run_study(n_sample=None, lang=None):
  np.random.seed(SEED_GLOBAL)

  optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS_PRUNING, n_warmup_steps=0, interval_steps=1, n_min_trials=1)  # https://optuna.readthedocs.io/en/stable/reference/pruners.html
  optuna_sampler = optuna.samplers.TPESampler(seed=SEED_GLOBAL, consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False, 
                                              n_startup_trials=N_STARTUP_TRIALS_SAMPLING, n_ei_candidates=24, multivariate=False, group=False, warn_independent_sampling=True, constant_liar=False)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
  study = optuna.create_study(direction="maximize", study_name=None, pruner=optuna_pruner, sampler=optuna_sampler)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html

  study.optimize(lambda trial: optuna_objective(trial, lang=lang, n_sample=n_sample, df_train=df_train, df=df_cl),  # hypothesis_hyperparams_dic=hypothesis_hyperparams_dic
                n_trials=N_TRIALS, show_progress_bar=True)  # Objective function with additional arguments https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
  return study


hp_study_dic = {}
for n_sample in N_SAMPLE_DEV:
    hp_study_dic_scenario = {}

    study = run_study(n_sample=n_sample)  #lang=lang

    print("Study params: ", LANGUAGE_LST, f" {VECTORIZER}_{MODEL_NAME.split('/')[-1]}_{DATASET}_{n_sample}")
    print("Best study value: ", study.best_value)
    hp_study_dic_lang = {"language_train": LANGUAGE_LST, "vectorizer": VECTORIZER, "augmentation": None, "method": METHOD, "n_sample": n_sample, "dataset": DATASET, "algorithm": MODEL_NAME, "nmt_model": NMT_MODEL, "optuna_study": study}
    hp_study_dic_scenario.update(hp_study_dic_lang)

    hp_study_dic_scenario = {f"{VECTORIZER}_{MODEL_NAME.split('/')[-1]}_{DATASET}_{n_sample}_{NMT_MODEL}": hp_study_dic_scenario}
    hp_study_dic.update(hp_study_dic_scenario)

    # harmonise length of n_sample string (always 5 characters)
    while len(str(n_sample)) <= 4:
        n_sample = "0" + str(n_sample)

    lang_str = "-".join(LANGUAGE_LST)

    ## save studies
    if EXECUTION_TERMINAL == True:
      joblib.dump(hp_study_dic_scenario, f"./{TRAINING_DIRECTORY}/optuna/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample}samp_{DATASET}_{lang_str}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}.pkl")
    elif EXECUTION_TERMINAL == False:
      joblib.dump(hp_study_dic_scenario, f"./{TRAINING_DIRECTORY}/optuna/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample}samp_{DATASET}_{lang_str}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}_t.pkl")


print("Run done.")

