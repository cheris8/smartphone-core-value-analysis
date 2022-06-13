import os
import torch
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/chaehyeong/nas/TextMiningProject')
    parser.add_argument('--training_size', type=float, default=0.8, help='Proportion of samples to use for training LM.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balancing', type=str, default='balancing', choices=['none', 'undersampling', 'oversampling', 'balancing'])
    args = parser.parse_args()
    args.base_dir = os.path.join(args.root_dir, args.balancing)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.training_size > 1:
        args.training_size = int(args.training_size) 
    return args

args = parser_args()