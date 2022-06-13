import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification, AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from settings import args
from dataset import  TextClassificationDataset
from dataset import load_coupang, prepare_batch
from utils import EarlyStopping
from utils import make_dir, train, evaluate

import warnings
import transformers
warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def main(args):
    make_dir(args.base_dir)
    print('Directory:', args.base_dir)
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    
    # load and split data
    target_names = ['0', '1', '2', '3', '4']

    texts, labels = load_coupang('train', args.balancing)
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(texts, labels, train_size=args.training_size, stratify=labels, random_state=args.seed)

    # create dataset and dataloader
    train_dataset = TextClassificationDataset(tokenizer, train_texts, train_labels)
    valid_dataset = TextClassificationDataset(tokenizer, valid_texts, valid_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: prepare_batch(x))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: prepare_batch(x))
    print('Train Set:', len(train_dataset), 'Valid Set:', len(valid_dataset))
    
    model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=len(target_names))
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch, last_epoch=-1, verbose=False)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # train
    for i in range(args.n_epoch):
        train_epoch_loss, train_labels, train_preds = train(model, optimizer, train_loader, args.device)
        valid_epoch_loss, valid_labels, valid_preds = evaluate(model, valid_loader, args.device)
        print(f"Epoch {i}")
        print(f"Train loss: {train_epoch_loss}")
        print(f"Valid loss: {valid_epoch_loss}")
        print("Train eval")
        print(classification_report(train_labels, train_preds, target_names=target_names))
        print("Valid eval")
        print(classification_report(valid_labels, valid_preds, target_names=target_names))

        valid_acc = accuracy_score(valid_labels, valid_preds)
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')
        model_name = 'ts{}-bs{}-lr{}-epoch{}-acc{:.04f}-f1{:.04f}.pt'.format(args.training_size, args.batch_size, args.lr, i+1, valid_acc, valid_f1)
        model_path = os.path.join(args.base_dir, model_name)
        early_stopping(valid_epoch_loss, model, model_path)
        if early_stopping.early_stop:
            print("Early stopping")              
            break
        scheduler.step()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main(args)

