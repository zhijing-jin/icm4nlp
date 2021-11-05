import os
from collections import namedtuple, defaultdict
import random

from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from easydict import EasyDict as edict

basepath = INSERT_YOUR_BASEPATH_HERE # Changed for anonymity. Write your dataset basepath here.

class TranslationDataset:
    def __init__(self, this_args):
        args = this_args
        if type(args) == dict:
            args = edict(args)
        self.device = args.device
        self.trim_length = args.trim_length
        self.source = args.source_lang
        self.target = args.target_lang
        self.direction_causal = args.direction_causal
        self.batch_size = args.batch_size
        self.confounder_lang = ""
        if 'confounder_lang' in this_args:
            self.confounder_lang = this_args['confounder_lang'] + "_"
        print("Confounder:", self.confounder_lang)

        if self.direction_causal:
            self.source_file_train = os.path.join(basepath, self.confounder_lang + self.source + '-' + self.target + '.' + self.source + '.train')
            self.source_file_dev   = os.path.join(basepath, self.confounder_lang + self.source + '-' + self.target + '.' + self.source + '.dev')
            self.target_file_train = os.path.join(basepath, self.confounder_lang + self.source + '-' + self.target + '.' + self.target + '.train')
            self.target_file_dev   = os.path.join(basepath, self.confounder_lang + self.source + '-' + self.target + '.' + self.target + '.dev')
        else:
            self.source_file_train = os.path.join(basepath, self.confounder_lang + self.target + '-' + self.source + '.' + self.source + '.train')
            self.source_file_dev   = os.path.join(basepath, self.confounder_lang + self.target + '-' + self.source + '.' + self.source + '.dev')
            self.target_file_train = os.path.join(basepath, self.confounder_lang + self.target + '-' + self.source + '.' + self.target + '.train')
            self.target_file_dev   = os.path.join(basepath, self.confounder_lang + self.target + '-' + self.source + '.' + self.target + '.dev')

        print(self.source, self.target, self.source_file_train, self.source_file_dev, self.target_file_train, self.target_file_dev)
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-"+self.source+'-'+self.target, use_fast=True)
        print("Loaded tokenizer")

        self.source_sentences_train = [x.strip()[:self.trim_length] for x in open(self.source_file_train).readlines()]
        self.source_sentences_dev   = [x.strip()[:self.trim_length] for x in open(self.source_file_dev).readlines()]
        self.target_sentences_train = [x.strip()[:self.trim_length] for x in open(self.target_file_train).readlines()]
        self.target_sentences_dev   = [x.strip()[:self.trim_length] for x in open(self.target_file_dev).readlines()]

        assert len(self.source_sentences_train) == len(self.target_sentences_train)
        assert len(self.source_sentences_dev)   == len(self.target_sentences_dev)
        print("Dataset Length (train, dev):", len(self.source_sentences_train), len(self.source_sentences_dev))

        if args.dummy:
            self.train_dataset = self.batch_dataset(self.source_sentences_train[:100], self.target_sentences_train[:100])
            self.dev_dataset   = self.train_dataset
        else:
            self.train_dataset = self.batch_dataset(self.source_sentences_train,
                                                    self.target_sentences_train
                                                )
            print("Dataset batched for train:", len(self.train_dataset))
            self.dev_dataset   = self.batch_dataset(self.source_sentences_dev,
                                                    self.target_sentences_dev
                                                )
            print("Dataset batched for dev:", len(self.dev_dataset))

    def batch_dataset(self, src, tgt):
        assert len(src) == len(tgt)
        data = [(s,t) for s,t in zip(src, tgt)]
        # data = sorted(data, key=lambda x: len(x[0]))
        batches = []
        idx = 0
        num_data = len(src)

        while idx < num_data:
            batch_src = [x[0] for x in data[idx:min(idx+self.batch_size, num_data)]]
            batch_tgt = [x[1] for x in data[idx:min(idx+self.batch_size, num_data)]]
            
            tokenized_src = self.tokenizer.batch_encode_plus(batch_src,
                                    padding=True, return_tensors="pt"
                                )
            with self.tokenizer.as_target_tokenizer():
                tokenized_tgt = self.tokenizer.batch_encode_plus(batch_tgt,
                                        padding=True, return_tensors="pt"
                                    )

            batches.append(({k: v.to(self.device) for k,v in tokenized_src.items()},
                        {k: v.to(self.device) for k,v in tokenized_tgt.items()}
                    ))
            idx += self.batch_size
        
        diff_subtoken = lambda x : abs(x[0]['attention_mask'].sum().cpu().detach().item() - x[1]['attention_mask'].sum().cpu().detach().item())
        sorted_batches = sorted(batches, key=diff_subtoken)
        print(diff_subtoken(sorted_batches[0]), diff_subtoken(sorted_batches[1]), diff_subtoken(sorted_batches[2]))
        return sorted_batches

class LMDataset:
    def __init__(self, this_args):
        args = this_args
        if type(args) == dict:
            args = edict(args)
        self.device = args.device
        self.trim_length = args.trim_length
        self.source = args.source_lang
        self.target = args.target_lang
        self.lm_lang = args.lm_lang
        if 'confounder_lang' in this_args:
            self.confounder_lang = this_args['confounder_lang'] + "_"
        print("Confounder:", self.confounder_lang)
        self.sents_file_train = os.path.join(basepath, self.confounder_lang + self.source + '-' + self.target + '.' + self.lm_lang + '.train')
        self.sents_file_dev   = os.path.join(basepath, self.confounder_lang + self.source + '-' + self.target + '.' + self.lm_lang + '.dev')
        self.batch_size = args.batch_size

        print(self.source, self.target, self.sents_file_train, self.sents_file_dev)
        if self.lm_lang == 'en':
            model_card = "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_card, use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.lm_lang == 'es':
            model_card = "datificate/gpt2-small-spanish"
            self.tokenizer = AutoTokenizer.from_pretrained(model_card, use_fast=True)
        elif self.lm_lang == 'fr':
            model_card = "dbddv01/gpt2-french-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_card, use_fast=True)
        else:
            raise "Language not supported for LM"

        print(model_card + " tokenizer has been loaded")

        self.sentences_train = [x.strip()[:self.trim_length]
                        for x in open(self.sents_file_train).readlines()]
        self.sentences_dev = [x.strip()[:self.trim_length]
                        for x in open(self.sents_file_dev).readlines()]

        print("Dataset Length (train, dev):", len(self.sentences_train), len(self.sentences_dev))

        if args.dummy:
            self.train_dataset = self.batch_dataset(self.sentences_train[:100])
            self.dev_dataset   = self.train_dataset
        else:
            self.train_dataset = self.batch_dataset(self.sentences_train)
            print("Dataset batched for train:", len(self.train_dataset))
            self.dev_dataset   = self.batch_dataset(self.sentences_dev)
            print("Dataset batched for dev:",   len(self.dev_dataset))

    def batch_dataset(self, sents):
        batches = []
        idx = 0
        num_data = len(sents)

        while idx < num_data:
            batch_sent = sents[idx:min(idx+self.batch_size, num_data)]
            tokenized_sent = self.tokenizer.batch_encode_plus(batch_sent,
                                    padding=True, return_tensors="pt"
                                )

            source_batch = {k: v.to(self.device) for k,v in tokenized_sent.items()}
            batches.append((source_batch, source_batch))
            idx += self.batch_size
        return batches


if __name__ == "__main__":
    class test_args:
        def __init__(self):
            self.source_lang = 'fr'
            self.target_lang = 'es'
            self.direction_causal = True
            self.batch_size = 64
            self.dummy = False
            self.trim_length = 400
            self.device = 'cuda:0'
            self.lm_lang = 'fr'
    
    # td = TranslationDataset(test_args())
    # print(len(td.train_dataset), len(td.dev_dataset))
    # print(td.train_dataset[0])
    # print(td.dev_dataset[0])
    # import os
    # os.system('nvidia-smi')

    lmd = LMDataset(test_args())
    print(len(lmd.train_dataset), len(lmd.dev_dataset))
    print(lmd.train_dataset[0])
    print(lmd.dev_dataset[0])
    import os
    os.system('nvidia-smi')
