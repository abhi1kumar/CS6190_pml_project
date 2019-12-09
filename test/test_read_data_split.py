

import numpy as np

def test_data_sanity(X, Y, idx):

    print("X shape : ", X.shape,"\nY shape : ", Y.shape, "\n")
    feat_index = np.arange(X.shape[1])
    label_index = np.arange(Y.shape[1])
    print("Sparse representation of X[",idx,"] : \n", feat_index[X[idx]==1],"\n")
    print("Sparse representation of Y[", idx, "] : \n", label_index[Y[idx]==1],"\n")
    print("Number of ones in X[{}] : {} \n".format(idx, np.sum(X[idx])))
    print("Number of ones in Y[{}] : {} \n".format(idx, np.sum(Y[idx])))
