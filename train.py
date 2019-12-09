

"""
    Sample Run:
    python train.py

    Start training the framework. 
 
"""
import argparse
import numpy as np

from util import *
from EM_Algorithm import *
from test.test_read_data_split import *

parser = argparse.ArgumentParser(description='Read XML data from files')
parser.add_argument('--dir_path'            , type=str, default='data/Bibtex'        , help='path of the directory containing data-file')
parser.add_argument('--data_filename'       , type=str, default='Bibtex_data.txt'    , help='rel path of data_filename')
parser.add_argument('--train_split_filename', type=str, default='bibtex_trSplit.txt' , help='rel path of train_split_filename')
parser.add_argument('--test_split_filename' , type=str, default='bibtex_tstSplit.txt', help='rel path of test_split_filename')
args = parser.parse_args()

# Get the data and the labels on the dataset
X, Y = get_data_labels_from_datafile(args.dir_path, args.data_filename)

train_indices_for_each_split = get_indices_for_different_splits(args.dir_path, args.train_split_filename)
test_indices_for_each_split  = get_indices_for_different_splits(args.dir_path, args.test_split_filename)
num_splits = train_indices_for_each_split.shape[0]

# Get some stats
print("")
print("Number of splits             = {}".format(num_splits))
print("Train indices for each split =", train_indices_for_each_split.shape)
print(" Test indices for each split =", test_indices_for_each_split.shape)

num_iterations = 1

for split in range(num_splits):
    train_X = X[split, :]
    train_Y = Y[split, :]
    EM_algorithm(num_iterations, train_X, train_Y, V, U, M, W, beta, lambda_u, lambda_v, lambda_beta, lambda_w, r)
