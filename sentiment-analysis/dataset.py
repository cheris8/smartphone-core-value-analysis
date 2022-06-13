import pandas as pd
import torch
from torch.utils.data import Dataset

def load_coupang(mode, balancing):
    if mode == 'test':
        file = pd.read_csv('/home/chaehyeong/TextMiningProject/testset.csv')
    else:
        if balancing == 'none':
            file = pd.read_csv('/home/chaehyeong/TextMiningProject/trainingset.csv')
        elif balancing == 'undersampling':
            file = pd.read_csv('/home/chaehyeong/TextMiningProject/trainingset_undersampling.csv')
        elif balancing == 'oversampling':
            file = pd.read_csv('/home/chaehyeong/TextMiningProject/trainingset_oversampling.csv')
        elif balancing == 'balancing':
            file = pd.read_csv('/home/chaehyeong/TextMiningProject/trainingset_balancing.csv')
    file.dropna(inplace=True)
    texts = file['상품평'].tolist()
    labels = file['별점'].tolist()
    labels = [ele-1 for ele in labels]
    return texts, labels


class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.tokenized_texts = []
        for text in self.texts:
            tokenized = self.tokenizer.encode_plus(text, max_length=512, truncation=True, padding=True, return_tensors='pt')
            self.tokenized_texts.append(tokenized)
            
    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = self.tokenized_texts[idx]['input_ids'].view(-1)
        item['attention_mask'] = self.tokenized_texts[idx]['attention_mask'].view(-1)
        item['token_type_ids'] = self.tokenized_texts[idx]['token_type_ids'].view(-1)
        item['labels'] = torch.tensor([self.labels[idx]]).view(-1)
        return item

    def __len__(self):
        return len(self.labels)


def prepare_batch(batch):
    batch_size = len(batch)

    input_ids_features = []
    token_type_ids_features = []
    attention_mask_features = []
    labels = []
    
    # flatten
    max_len = 0
    for b in batch:
        input_ids_features.append(b['input_ids'])
        token_type_ids_features.append(b['token_type_ids'])
        attention_mask_features.append(b['attention_mask'])
        labels.append(b['labels'])
        if b['input_ids'].shape[0] > max_len:
            max_len = b['input_ids'].shape[0]
    
    # padding
    padded_input_ids_features = []
    padded_token_type_ids_features = []
    padded_attention_mask_features = []
    for input_ids, token_type_ids, attention_mask in zip(input_ids_features, token_type_ids_features, attention_mask_features):
        pad_len = max_len - input_ids.shape[0]
        if pad_len > 0:
            padded_input_ids = torch.cat([input_ids, torch.LongTensor([0] * pad_len)])
            padded_token_type_ids = torch.cat([token_type_ids, torch.LongTensor([0] * pad_len)])
            padded_attention_mask = torch.cat([attention_mask, torch.LongTensor([0] * pad_len)])
            padded_input_ids_features.append(padded_input_ids)
            padded_token_type_ids_features.append(padded_token_type_ids)
            padded_attention_mask_features.append(padded_attention_mask)
        else:
            padded_input_ids_features.append(input_ids)
            padded_token_type_ids_features.append(token_type_ids)
            padded_attention_mask_features.append(attention_mask)
    # un-flatten
    batch = {}
    batch['input_ids'] = torch.stack(padded_input_ids_features).view(batch_size, -1)
    batch['token_type_ids'] = torch.stack(padded_token_type_ids_features).view(batch_size, -1)
    batch['attention_mask'] = torch.stack(padded_attention_mask_features).view(batch_size, -1)
    batch['labels'] = torch.stack(labels).view(batch_size, -1)
    return batch