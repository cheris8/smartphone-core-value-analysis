import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification

from sklearn.metrics import classification_report

from settings import args
from dataset import  TextClassificationDataset
from dataset import load_coupang
from utils import get_best_model_path, test

import warnings
import transformers
warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def main(args):
    print('Directory:', args.base_dir)
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    
    # load and split data
    target_names = ['1', '2', '3', '4', '5']

    test_texts, test_labels = load_coupang('test', args.balancing)

    # create dataset and dataloader
    test_dataset = TextClassificationDataset(tokenizer, test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print('Test Set:', len(test_dataset))
    
    # evaluate
    model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=len(target_names))
    best_path = get_best_model_path(args.base_dir)
    restore_dict = torch.load(best_path, map_location=args.device)
    model.load_state_dict(restore_dict)
    print('Best model Path:', best_path)
    test_labels, test_preds, test_logits = test(model, test_loader, args.device)
    print(classification_report(test_labels, test_preds, target_names=target_names))
    with open(f'{best_path}_metrics.txt', 'w') as f:
        f.write(classification_report(test_labels, test_preds, target_names=target_names))
    
    # save test results
    results = pd.DataFrame({'texts':test_texts, 'labels':test_labels, 'preds':test_preds, 'logits':test_logits})
    results.to_csv(f'{best_path}_results.csv')

if __name__ == '__main__':
    main(args)