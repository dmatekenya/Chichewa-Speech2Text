from datasets import load_dataset, load_metric, Dataset, DatasetDict
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import wavfile
import random
from IPython.display import display, HTML
import IPython.display as ipd
import re
import shutil
from sphfile import SPHFile
import json
from transformers import Wav2Vec2FeatureExtractor,  Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# Global variables
CWD = Path.cwd().parents[0]
DIR_DATA = CWD.joinpath('data', 'TIMIT')
DIR_TRAIN = DIR_DATA.joinpath('TRAIN')
DIR_TEST = DIR_DATA.joinpath('TEST')
FILE_METADATA_TRAIN = DIR_DATA.joinpath('train_metadata.csv')
FILE_METADATA_TEST = DIR_DATA.joinpath('test_metadata.csv')
GENERATE_DATA = False
NUM_THREADS = 28
torch.set_num_threads(NUM_THREADS)


def remove_special_charecters(txt):
    chars_to_ignore_regex = '[\,\),\?\.\!\-\;\:\"]'
    lower_txt = txt.lower()

    txt2 = re.sub(chars_to_ignore_regex, '', lower_txt)
    return txt2


def convert_files_to_wav(df):
    for idx, row in df.iterrows():
        try:
            curr_fpath = Path(row['aud_file'])
            fname = curr_fpath.parts[-1].split(".")[0]
            new_fpath = curr_fpath.joinpath(curr_fpath.parents[0], "{}_wav.wav".format(fname))
            sph = SPHFile(str(curr_fpath))
            sph.write_wav(filename = str(new_fpath))
            df.loc[idx, 'aud_file'] = str(new_fpath)
        except:
            print('Something went wrong')
    return df

def convert_weird_audio_format2wav():
    df = convert_files_to_wav(df=df_test)
    df.to_csv(FILE_METADATA_TEST, index=False)
    
    df = convert_files_to_wav(df=df_train)
    df.to_csv(FILE_METADATA_TRAIN, index=False)

def extract_all_chars(df, col):
    text_vals = list(df[col].values)
    all_text = " ".join(text_vals)
    vocab = list(set(all_text))
    return {"vocab": vocab, "all_text": all_text}


def convert_files_to_wav(ind):
    for idx, row in df.iterrows():
        try:
            curr_fpath = Path(row['aud_file'])
            fname = curr_fpath.parts[-1].split(".")[0]
            new_fpath = curr_fpath.joinpath(curr_fpath.parents[0], "{}_wav.wav".format(fname))
            sph = SPHFile(str(curr_fpath))
            sph.write_wav(filename = str(new_fpath))
            df.loc[idx, 'aud_file'] = str(new_fpath)
        except:
            print('Something went wrong')
    return df


def show_random_elements(df, num_examples=10):
    assert num_examples <= len(df), "Can't pick more elements than there are in the dataset."
    indices = df.index.values
    picks = random.choices(indices, k=num_examples)
    df = df.iloc[picks][['text']]
    display(HTML(df.to_html()))

def generate_vocubalary(df_train, df_test, output_vocab_file):
    res_train = extract_all_chars(df_train, "text_no_quotes")
    res_test = extract_all_chars(df_test, "text_no_quotes")
    
    vocab_list = list(set(res_train["vocab"]) | set(res_test["vocab"]))
    
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    

    with open(vocab_file, 'w') as file:
        json.dump(vocab_dict, file)

