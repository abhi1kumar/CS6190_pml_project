

import argparse
import torch
import os.path as osp
import numpy as np

def get_data_labels_from_datafile(dir_path, data_file_rel_path):
    """
        Reads data file
    """
    data_file_full_path = osp.join(dir_path, data_file_rel_path)
    print("Reading {}...".format(data_file_full_path))

    with open(data_file_full_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if(idx==0):
                N, D, L = [int(k) for k in line.split(' ')]
                X = np.zeros((N, D))
                Y = np.zeros((N, L))
            else:
                temp = line.split(' ')
                for y_idx in temp[0].split(','):
                    Y[idx-1][int(y_idx)] = 1
                for xval_info in temp[1:]:
                    x_idx, x_val = xval_info.split(':')
                    X[idx-1][int(x_idx)] = float(x_val)
    print("Data   X shape =", X.shape)
    print("Labels Y shape =", Y.shape)

    return X, Y

def get_indices_for_different_splits(dir_path, split_file_rel_path):
    """
        Returns arrays of each of the training splits.
        The indices are in the columns of the file.
    """
    split_file_full_path = osp.join(dir_path, split_file_rel_path)
    print("Reading {}...".format(split_file_full_path))
    
    indices_for_each_split = np.genfromtxt(split_file_full_path, delimiter= " ").T
    # Indices should be zero indexed
    indices_for_each_split  -= 1

    return indices_for_each_split

def precision_at_k(X, Y, W, beta, psi, topk):
    """
            Calculates the metric precision@k

            Inputs:
            X    = shape N  x D
            Y    = shape N  x Ls
            V    = shape Ls x K
            W    = shape D  x K
            beta = shape L  x K
            psi  = shape K  x K

            k = top-k accuracy
    """

    U = torch.matmul(X, W)          # U = shape N x K
    V = torch.matmul(beta, psi)     # V = shape L x K
    y_pred = torch.matmul(U, V.t()) #y_pred = shape N x L
    y_pred_sort_idx = torch.argsort(y_pred, dim = 1, descending=True)[:,:topk]
    y_pred_sort_idx = y_pred_sort_idx + (y_pred.shape[1]*torch.arange(y_pred.shape[0]))[:, None]
    y_pred_sort_idx_f = y_pred_sort_idx.flatten()
    Y_f = Y.flatten()
    precision = torch.sum(Y_f[y_pred_sort_idx_f]==1)/(Y.shape[0]*topk)
    return precision
