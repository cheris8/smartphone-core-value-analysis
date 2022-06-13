import os
import pathlib
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_best_model_path(path):
    all_files = os.listdir(path)
    files = [file for file in all_files if file.endswith('.pt')]
    accuracies = [float(file[-9:-3]) for file in files]
    assert len(files) == len(accuracies)
    acc_to_file = {}
    for file, acc in zip(files, accuracies):
        acc_to_file[acc] = file
    best_acc = max(acc_to_file.keys())
    return os.path.join(path, acc_to_file.get(best_acc))


def train(model, optimizer, loader, device):
    model.to(device)
    model.train() # Run model in training mode
    epoch_true_labels = []
    epoch_preds = []
    epoch_loss = 0
    for batch in tqdm(loader):
        # input_ids shape: (batch_size, sequence_length)
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # labels shape: (batch_size, )
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs[0], outputs[1]
        preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        
        epoch_true_labels.extend(labels.tolist())
        epoch_preds.extend(preds.tolist())
        epoch_loss += loss.item()
        
        # back propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.empty_cache()
    return epoch_loss / len(loader), epoch_true_labels, epoch_preds


def evaluate(model, loader, device):
    model.to(device)
    model.eval() # Run model in eval mode (disables dropout layer)
    epoch_true_labels = []
    epoch_preds = []
    epoch_loss = 0
    with torch.no_grad(): # Disable gradient computation - required only during training
        for batch in tqdm(loader):
            # input_ids shape: (batch_size, sequence_length)
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels shape: (batch_size, )
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[0], outputs[1]
            preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            
            epoch_true_labels.extend(labels.tolist())
            epoch_preds.extend(preds.tolist())
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
    return epoch_loss/len(loader), epoch_true_labels, epoch_preds


def test(model, loader, device):
    """
    Evaluate the model on a test set.
    Only do batch size = 1.
    """
    epoch_true_labels = []
    epoch_preds = []
    epoch_logits = []

    model = model.to(device)
    model.eval()
    with torch.no_grad(): 
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            _, logits = outputs[0], outputs[1]
            preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)

            epoch_true_labels.append(int(labels))
            epoch_preds.extend(preds.tolist())
            epoch_logits.extend(logits.tolist())

            torch.cuda.empty_cache()

    return epoch_true_labels, epoch_preds, epoch_logits


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            trace_func (function): trace print function. Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss