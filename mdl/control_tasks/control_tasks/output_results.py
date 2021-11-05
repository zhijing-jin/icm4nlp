import json
import torch
from transformers import AutoTokenizer
import os
import numpy as np
from argparse import ArgumentParser
from easydict import EasyDict as edict

argp = ArgumentParser()
argp.add_argument('experiment_config')
cli_args = argp.parse_args()

saves_dir = os.path.join('saves', cli_args.experiment_config.split("/")[-1].split('.')[0])
config_args = json.load(open(os.path.join(saves_dir, 'args.json')))
if 'lm_lang' not in config_args.keys():
    config_args = edict(config_args)
    model_card = "Helsinki-NLP/opus-mt-" + config_args.source_lang + '-' + config_args.target_lang
else:
    config_args = edict(config_args)
    if config_args.lm_lang == 'en':
        model_card = "gpt2"
    elif config_args.lm_lang == 'es':
        model_card = "datificate/gpt2-small-spanish"
    elif config_args.lm_lang == 'fr':
        model_card = "dbddv01/gpt2-french-small"
    else:
        raise "Language not supported for LM"

tokenizer = AutoTokenizer.from_pretrained(model_card, use_fast=True)
num_classes = tokenizer.vocab_size
print("Vocab size:", num_classes)
online_report = json.load(open(os.path.join(saves_dir, 'online_coding.json')))

# Uniform codelength
total_train_size = sum(x['train_targets'][0] for x in online_report) + online_report[-1]['num_targets_words_online_portion']
uniform_codelength = total_train_size * np.log2(num_classes)

# Online codelength
print("Total number of subwords in t1:", online_report[0]['train_targets'][0])
print(online_report[0]['train_targets'][0] * np.log2(num_classes))
online_codelength = online_report[0]['train_targets'][0] * np.log2(num_classes) + sum(elem['loss_online_portion'] for elem in online_report)

print("Uniform codelength: {} kbits".format(round(uniform_codelength / 1024, 2)))
print("Online codelength: {} kbits".format(round(online_codelength / 1024, 2)))
print("Compression: {} ".format(round(uniform_codelength / online_codelength, 2)))
