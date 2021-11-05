"""Loads configuration yaml and runs an experiment."""
from argparse import ArgumentParser
import os
import json, pickle
import random
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import random
import numpy as np

import data
import probe
import regimen
import task
import loss

from torch.utils.data import DataLoader, Dataset

def choose_regimen_class(args):
    """
        Chooses regimen.

        Args:
          args: the global config dictionary built by yaml.
        Returns:
          A class to be as regimen.
    """
    if not 'type' in args['regimen'] or args['regimen']['type'] == 'none':
        return regimen.ProbeRegimen
    elif args['regimen']['type'] == 'bayes':
        return regimen.BayesRegimen
    elif args['regimen']['type'] == 'online_coding':
        return regimen.ProbeRegimen
    else:
        raise ValueError("Unknown regimen type: {}".format(
                        args['regimen']['type']))

def execute_experiment(args, train_probe):
    """
        Execute an experiment as determined by the configuration
        in args.

        Args:
          train_probe: Boolean whether to train the probe
    """
    regimen_class = choose_regimen_class(args) # regimen.ProbeRegimen

    if 'lm_lang' not in args.keys():
        print("+++++Running a translation model.+++++")
        expt_dataset = data.TranslationDataset(args)
        probe_class = probe.TranslationModel
    else:
        print("+++++Running a language model.+++++")
        expt_dataset = data.LMDataset(args)
        probe_class = probe.LMModel
    expt_probe = probe_class(args)
    expt_model = lambda x: x
    expt_regimen = regimen_class(args)
    expt_loss = loss.SequenceCrossEntropyLoss(args)

    def split_data_into_portions(dataset_train_dataset):
        total_len = len(dataset_train_dataset)
        fractions = list(map(float, args['regimen']['inds'].split(',')))

        train_portions = []
        eval_portions = []
        for i in range(len(fractions)):
            train_portions.append(dataset_train_dataset[: max(1,int(fractions[i] * total_len))])
            if i != len(fractions) - 1:
                eval_portions.append(dataset_train_dataset[int(fractions[i] * total_len):
                                                           max(int(fractions[i] * total_len) + 1, int(fractions[i + 1] * total_len))
                                                        ])
        print("Dataset Portion Stats:", [len(tp) for tp in train_portions],
                [len(tp) for tp in eval_portions])
        return train_portions, eval_portions

    if args['regimen']['type'] != 'online_coding':
        raise NotImplementedError
    else:
        online_coding_list = []
        dev_dataloader = expt_dataset.dev_dataset

        # print("\n\nShuffling dataset with seed {}!!!\n\n".format(args["seed"]))
        shuffled_dataset = expt_dataset.train_dataset
        # random.shuffle(shuffled_dataset)
        train_portions, eval_portions = split_data_into_portions(shuffled_dataset)
        for i in range(len(train_portions) - 1):
            print("==============")
            print("+++", i, "+++")
            print("==============")
            
            expt_probe = probe_class(args)
            current_train = train_portions[i]
            current_dev = eval_portions[i]

            # run-train-probe
            _, evals = expt_regimen.train_until_convergence(
                                          expt_probe, expt_model, expt_loss,
                                          current_train, dev_dataloader, 
                                          {'online_portion': current_dev}, i
                                      )
            online_coding_list.append(evals)
            print(evals)
            json.dump(online_coding_list, open(os.path.join(args['target_folder'],
                                            'online_coding'+str(i)+'.json'), 'w+'))

        # save results
        json.dump(online_coding_list, open(os.path.join(args['target_folder'], 'online_coding.json'), 'w+'))

if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--results-dir', default='',
        help='Set to reuse an old results dir; '
        'if left empty, new directory is created')
    argp.add_argument('--train-probe', default=-1, type=int,
        help='Set to train a new probe.; ')
    argp.add_argument('--report-results', default=1, type=int,
        help='Set to report results; '
        '(optionally after training a new probe)')
    argp.add_argument('--embeddings-path', default='',
        help='sets all random seeds for (within-machine) reproducibility')
    argp.add_argument('--seed', default=4321, type=int,
        help='sets all random seeds for (within-machine) reproducibility')
    cli_args = argp.parse_args()
    if cli_args.seed:
        random.seed(cli_args.seed)
        np.random.seed(cli_args.seed)
        torch.manual_seed(cli_args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    yaml_args= yaml.load(open(cli_args.experiment_config))
    yaml_args['seed'] = cli_args.seed
    yaml_args['target_folder'] = 'saves/'+cli_args.experiment_config.split("/")[-1].split('.')[0]
    os.makedirs(yaml_args['target_folder'], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yaml_args['device'] = device
    print(yaml_args)
    json.dump(yaml_args, open(yaml_args['target_folder'] + '/args.json', 'w+'))
    execute_experiment(yaml_args, train_probe=cli_args.train_probe)
