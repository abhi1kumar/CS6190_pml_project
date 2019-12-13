

import argparse
import torch
import os.path as osp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def normaliseX(X):
    """
        Normalise the matrix X
    """
    X_l2   = torch.sqrt(torch.sum(torch.pow(X,2), axis=1))[:, None]
    X_norm = X_norm/X_l2
    return X_norm

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

def eval_precision(U, V, Y, topk):

    val = [1, 3, 5]
    prec = []
    y_pred = torch.matmul(U, V.t()) #y_pred = shape N x L
    y_pred_sort_idx = torch.argsort(y_pred, dim = 1, descending=True)[:,:topk]
    y_pred_sort_idx = y_pred_sort_idx + (y_pred.shape[1]*torch.arange(y_pred.shape[0]))[:, None].to(device)
    for i in val:
        y_pred_sort_idx_f = y_pred_sort_idx[:,:i].flatten()
        Y_f = Y.flatten()
        precision = (1.0*torch.sum(Y_f[y_pred_sort_idx_f]==1))/(Y.shape[0]*i)
        prec.append(precision)
    return prec

def precision_at_k(X, Y, W, beta, psi, V, topk, num_seen_labels, isSeen = False, isUnseen = False, isSeenUnseen = False):
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
    #print(X.shape, Y.shape)

    U = torch.matmul(X, W)          # U = shape N x K
    if(isSeen):
        prec = eval_precision(U, V, Y, topk)
    if(isUnseen):
        V_unseen = torch.matmul(beta[num_seen_labels:,:], psi)
        prec = eval_precision(U, V_unseen, Y[:, num_seen_labels:], topk)
    if(isSeenUnseen):
        V_new = torch.matmul(beta, psi)
        prec = eval_precision(U, V_new, Y, topk)
    return prec
