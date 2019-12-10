

import os, sys
sys.path.append(os.getcwd())

import argparse
import numpy as np

import torch

from util import *
from EM_Algorithm import *

parser = argparse.ArgumentParser(description='Read XML data from files')
parser.add_argument('--dir_path'            , type=str, default='data/Bibtex'        , help='path of the directory containing data-file')
parser.add_argument('--data_filename'       , type=str, default='Bibtex_data.txt'    , help='rel path of data_filename')
parser.add_argument('--train_split_filename', type=str, default='bibtex_trSplit.txt' , help='rel path of train_split_filename')
parser.add_argument('--test_split_filename' , type=str, default='bibtex_tstSplit.txt', help='rel path of test_split_filename')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
topk = 5
K    = 500
split_idx = 0
num_seen_labels = 110

# Get the data and the labels on the dataset
X, Y = get_data_labels_from_datafile(args.dir_path, args.data_filename)
X, Y = torch.tensor(X).float(), torch.tensor(Y).type(torch.LongTensor)

train_indices_for_each_split = get_indices_for_different_splits(args.dir_path, args.train_split_filename)
test_indices_for_each_split  = get_indices_for_different_splits(args.dir_path, args.test_split_filename)

train_indices = train_indices_for_each_split[split_idx]
test_indices  = test_indices_for_each_split [split_idx]

train_X      = X[train_indices]
train_Y_seen = Y[train_indices, :num_seen_labels]
train_Y      = Y[train_indices]

M = torch.matmul(train_Y.t(), train_Y)[:num_seen_labels,:].float().to(device) 

# Initialisation
V    = torch.ones  (train_Y.shape[1], K).float()
W    = torch.ones  (train_X.shape[1], K).float()
U    = torch.matmul(train_X, W)
beta = torch.ones  (M.shape[1], K).float()
psi  = torch.ones  (K, K).float()


print(train_X[0:2])
print(train_Y[0:2])
print("*******")

# Get Train precision
precision_train = precision_at_k(train_X, train_Y, W, beta, psi, topk)

print("Split index : {} \t Precision : {}".format(split_idx, precision_train))
