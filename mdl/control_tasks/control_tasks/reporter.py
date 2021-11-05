"""Contains classes for computing and reporting evaluation metrics."""

from collections import defaultdict
import os

from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
#from scipy.special import softmax
import numpy as np 
import json

import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

class Reporter:
    """Base class for reporting.

    Attributes:
      test_reporting_constraint: Any reporting method
        (identified by a string) not in this list will not
        be reported on for the test set.
    """

    def __init__(self, args, dataset):
        raise NotImplementedError("Inherit from this class and override __init__")

    def __call__(self, prediction_batches, dataloader, split_name, **kwargs):
        """
            Performs all reporting methods as specifed in the yaml experiment config dict.
            
            Any reporting method not in test_reporting_constraint will not
            be reported on for the test set.

            Args:
            prediction_batches: A sequence of batches of predictions for a data split
            dataloader: A DataLoader for a data split
            split_name the string naming the data split: {train,dev,test}
        """
        report = {}
        for method in self.reporting_methods:
            if method in self.reporting_method_dict:  
                report.update(self.reporting_method_dict[method](prediction_batches, dataloader, split_name, **kwargs))
                if not kwargs.get('is_train', False):
                    tqdm.write("Reporting {} on split {}".format(method, split_name))
                    self.reporting_method_dict[method](prediction_batches
                    , dataloader, split_name, **kwargs)
            else:
                tqdm.write('[WARNING] Reporting method not known: {}; skipping'.format(method))
        return report

    def write_json(self, prediction_batches, dataset, split_name, **kwargs):
        """
            Writes observations and predictions to disk.
            
            Args:
            prediction_batches: A sequence of batches of predictions for a data split
            dataset: A sequence of batches of Observations
            split_name the string naming the data split: {train,dev,test}
        """
        if not kwargs.get('is_train', False):
            json.dump([prediction_batch.tolist() for prediction_batch in prediction_batches]
                , open(os.path.join(self.reporting_root, split_name+'.predictions'), 'w'))
            json.dump([[x[0][:-1] for x in observation_batch] for _,_,_, observation_batch in dataset],
                open(os.path.join(self.reporting_root, split_name+'.observations'), 'w'))
        return {}

class TranslationReporter(Reporter):
    """Reporting class for single-word (depth) tasks"""

    def __init__(self, args, dataset):
        self.args = args
        self.reporting_methods = 'translation'
        self.reporting_method_dict = {
            'translation': self.report_translation
            }
        self.reporting_root = args['reporting']['root']
        self.test_reporting_constraint = {'spearmanr', 'uuas', 'root_acc'}
        self.dataset = dataset

    def report_label_values(self, prediction_batches, dataset, split_name, **kwargs):
        total = 0
        correct = 0
        for prediction_batch, (_, label_batch, length_batch, observation_batch) in zip(
              prediction_batches, dataset):
            for prediction, label, length in zip(
                  prediction_batch, label_batch,
                  length_batch):
                label = label[:length].cpu().numpy()
                predictions = np.argmax(prediction[:length], axis=-1)
                total += length.cpu().numpy()
                correct += np.sum(predictions == label)
            
        if not kwargs.get('is_train', False):
            with open(os.path.join(self.reporting_root, split_name + '.label_acc'), 'w') as fout:
                fout.write(str(float(correct)/  total) + '\n')
        return {'label_acc_{}'.format(split_name): float(correct)/  total}

