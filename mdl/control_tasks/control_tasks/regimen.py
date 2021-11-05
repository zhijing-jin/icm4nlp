"""Classes for training and running inference on probes."""
import os
import sys

from torch import optim
import torch
from tqdm import tqdm
import json

class ProbeRegimen:
    """
        Basic regimen for training and running inference on probes.
        
        Tutorial help from:
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Attributes:
          optimizer: the optimizer used to train the probe
          scheduler: the scheduler used to set the optimizer base learning rate
    """

    def __init__(self, args):
        self.args = args

        self.reports = []
        
        self.max_epochs = args['probe_training']['epochs']
        self.params_path = os.path.join(args['target_folder'], 'probe.pt')
        self.max_gradient_steps = args['probe_training']['max_gradient_steps'] if 'max_gradient_steps' in args['probe_training'] else sys.maxsize
        self.dev_eval_gradient_steps = args['probe_training']['eval_dev_every'] if 'eval_dev_every' in args['probe_training'] else -1

    def set_optimizer(self, probe):
        """
            Sets the optimizer and scheduler for the training regimen.
        
            Args:
            probe: the probe PyTorch model the optimizer should act on.
        """
        if 'weight_decay' in self.args['probe_training']:
            weight_decay = self.args['probe_training']['weight_decay']
        else:
            weight_decay = 0
        if 'scheduler_patience' in self.args['probe_training']:
            scheduler_patience = self.args['probe_training']['scheduler_patience']
        else:
            scheduler_patience = 0
        
        learning_rate = 0.001 if not 'learning_rate' in self.args['probe_training'] else\
                        self.args['probe_training']['learning_rate']
            
        scheduler_factor = 0.5 if not 'scheduler_factor' in self.args['probe_training'] else\
                        self.args['probe_training']['scheduler_factor']

        self.optimizer = optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode='min',
                                                            factor=scheduler_factor,
                                                            patience=scheduler_patience)

    def train_until_convergence(self, probe, model, loss, train_dataset, dev_dataset, eval_datasets, train_id):
        """
            Trains a probe until a convergence criterion is met.

            Trains until loss on the development set does not improve by more than epsilon
            for 5 straight epochs.

            Writes parameters of the probe to disk, at the location specified by config.

            Args:
              probe: An instance of probe.Probe, transforming model outputs to predictions
              model: An instance of model.Model, transforming inputs to word reprs
              loss: An instance of loss.Loss, computing loss between predictions and labels
              train_dataset: a torch.DataLoader object for iterating through training data
              dev_dataset: a torch.DataLoader object for iterating through dev data
        """
        train_id = str(train_id)
        def loss_on_dataset(dataset, name=''):
            loss_ = 0
            num_targets_sents = 0
            num_targets_words = 0
            num_examples = 0
            for batch in tqdm(dataset, desc='[eval batch{}]'.format(' ' + name)):
                source_batch, target_batch = batch
                batch_loss, count_0 = probe(source_batch, target_batch)
                count_1 = target_batch['input_ids'].shape[0] # batch size

                loss_ += batch_loss.sum().detach().cpu().numpy()
                num_targets_sents += count_1
                num_targets_words += count_0.sum()
            return {'loss{}'.format('_' + name): float(loss_),
                    'num_targets_sents{}'.format('_' + name): int(num_targets_sents),
                    'num_targets_words{}'.format('_' + name): int(num_targets_words),
                }

        def num_targets(dataset):
            num_targets = 0
            num_sents = 0
            for batch in tqdm(dataset):
                _, target_batch = batch
                num_targets += int(target_batch['attention_mask'].sum().detach().cpu().numpy())
                num_sents += int(target_batch['attention_mask'].shape[0])
            return num_targets, num_sents

        def eval_on_exit():
            probe.load_state_dict(torch.load(self.params_path))
            probe.eval()
            print("Evaling on exit...")
            result = {'train_targets': num_targets(train_dataset)}
            for name, dataset in eval_datasets.items():
                result.update(loss_on_dataset(dataset, name=name))
            return result
        
        self.set_optimizer(probe)
        min_dev_loss = sys.maxsize
        min_epoch_dev_loss = sys.maxsize
        min_dev_loss_epoch = -1
        gradient_steps = 0
        eval_dev_every = self.dev_eval_gradient_steps if self.dev_eval_gradient_steps != -1 else (len(train_dataset))
        eval_index = 0
        min_dev_loss_eval_index = -1
        
        self.wait_without_improvement_for = self.args['probe_training'].get('wait_without_improvement_for', 4)
        
        if not os.path.exists(os.path.join(self.args['target_folder'], 'checkpoint')):
            os.mkdir(os.path.join(self.args['target_folder'], 'checkpoint'))  

        for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
            epoch_train_loss = 0
            epoch_train_epoch_count = 0
            epoch_dev_epoch_count = 0
            epoch_train_loss_count = 0
            for batch in tqdm(train_dataset, desc='[training batch]'):
                probe.train()
                self.optimizer.zero_grad()
                source_batch, target_batch = batch
                batch_loss, count = probe(source_batch, target_batch)

                batch_loss = batch_loss.sum()
                count = count.sum().detach().cpu().numpy()
                batch_loss.backward()

                epoch_train_loss += batch_loss.detach().cpu().numpy()
                epoch_train_epoch_count += 1
                epoch_train_loss_count += count
                self.optimizer.step()
                gradient_steps += 1
                if gradient_steps % eval_dev_every == 0:
                    eval_index += 1
                    if gradient_steps >= self.max_gradient_steps:
                        tqdm.write('Hit max gradient steps; stopping')
                        with open(os.path.join(self.args['target_folder'], 'train_report'+train_id+'.json'), 'w+') as f:
                            json.dump(self.reports, f)
                        return self.reports, eval_on_exit()
                    epoch_dev_loss = 0
                    epoch_dev_loss_count = 0
                    for batch in tqdm(dev_dataset, desc='[dev batch]'):
                        self.optimizer.zero_grad()
                        probe.eval()
                        source_batch, target_batch = batch
                        batch_loss, count = probe(source_batch, target_batch)
                        batch_loss = batch_loss.sum()
                        count = count.sum().detach().cpu().numpy()
                        
                        epoch_dev_loss += batch_loss.detach().cpu().numpy()
                        epoch_dev_epoch_count += 1
                        epoch_dev_loss_count += count
                    self.scheduler.step(epoch_dev_loss)
                    tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index,
                        epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))

                    current_report = {'eval_step': eval_index,
                                      'gradient_steps': gradient_steps,
                                      'dev_loss': epoch_dev_loss/epoch_dev_loss_count,
                                      'train_loss': epoch_train_loss/epoch_train_loss_count,
                                      'min_dev_loss': min_dev_loss,
                                    'min_epoch_dev_loss': min_epoch_dev_loss}
                    self.reports.append(current_report)
                    # if True:
                    #     torch.save(probe.state_dict(),
                    #                 self.args['target_folder'] + '/checkpoint/probe_state_dict_{}'.format(epoch_index))
                    #     torch.save(probe,
                    #                 self.args['target_folder'] + '/checkpoint/probe_{}.pth'.format(epoch_index))
                    with open(os.path.join(self.args['target_folder'], 'train_report.json'), 'w+') as f:
                        json.dump(self.reports, f)
                      
                    if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.001:
                        torch.save(probe.state_dict(), self.params_path)
                        torch.save(probe, self.params_path + '_whole_probe.pth')
                        min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
                        min_epoch_dev_loss = epoch_dev_loss
                        min_dev_loss_epoch = epoch_index
                        min_dev_loss_eval_index = eval_index
                        tqdm.write('Saving probe parameters')
                    elif min_dev_loss_eval_index < eval_index - self.wait_without_improvement_for:
                        tqdm.write('Early stopping')
                        with open(os.path.join(self.args['target_folder'], 'train_report'+train_id+'.json'), 'w') as f:
                            json.dump(self.reports, f)
                        return self.reports, eval_on_exit()
        with open(os.path.join(self.args['target_folder'], 'train_report'+train_id+'.json'), 'w') as f:
            json.dump(self.reports, f)
        return self.reports, eval_on_exit()
