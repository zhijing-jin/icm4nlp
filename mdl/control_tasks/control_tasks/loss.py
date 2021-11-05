"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn
from tqdm import tqdm

class SequenceCrossEntropyLoss(nn.Module):
    """Custom cross-entropy loss"""
    def __init__(self,args=None):
        super(SequenceCrossEntropyLoss, self).__init__()
        tqdm.write('Constructing CrossEntropyLoss')
        # self.args = args
        self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1,
                                                        reduction='none')

    def forward(self, predictions, label_batch):
        """
            predictions: instance of transformers.modeling_outputs.Seq2SeqLMOutput
            label_batch: output of tokenizers.batch_encode plus for target labels (transformers.tokenization_utils_base.BatchEncoding)
        """
        loss = (self.pytorch_ce_loss(predictions.logits.permute(0, 2, 1),
                                    label_batch['input_ids']) * label_batch['attention_mask'])
        return loss, label_batch['attention_mask'].detach()#, predictions.logits.shape[0]

