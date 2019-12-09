import argparse
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser(description='Read XML data from files')
parser.add_argument('--dir_path', type=str, default='/home/aish/project/XML/pml/Bibtex', help='path of the directory containing data-file')
parser.add_argument('--data_filename', type=str, default='Bibtex_data.txt', help='data_filename')
parser.add_argument('--train_split_filename', type=str, default='bibtex_trSplit.txt', help='train_split_filename')
parser.add_argument('--test_split_filename', type=str, default='bibtex_tstSplit.txt', help='test_split_filename')
args = parser.parse_args()

def read_datafile(dir_path, data_filename):
	with open(osp.join(dir_path, data_filename), 'r') as f:
		for idx, line in enumerate(f.readlines()):
			if(idx==0):
				N, D, L = [int(k) for k in line.split(' ')]
				print(N, D, L)
				X = np.zeros((N, D))
				Y = np.zeros((N, L))
			else:
				temp = line.split(' ')
				for y_idx in temp[0].split(','):
					Y[idx-1][int(y_idx)] = 1
				for xval_info in temp[1:]:
					x_idx, x_val = xval_info.split(':')
					X[idx-1][int(x_idx)] = float(x_val)
	return X, Y

def read_splitfile(dir_path, filename):

	split = np.array([])
	with open(osp.join(dir_path, filename), 'r') as f:
		for line in f.readlines():
			temp = [int(k) for k in line.split(' ')]
			if(split.size==0):
				split = np.array(temp).reshape(-1,1)
			else :
				#print(split.shape)
				temp = np.array(temp).reshape(-1,1)
				split = np.hstack((split,temp))
	#print(split[:,0])
	#print(split.shape)
	return (split-1)	#for 0 indexing

def train_model(X, Y, split):

	train_X = X[split,:]
	train_Y = Y[split,:]
	EM_algorithm(iterations, train_X, train_Y V, U, M, W, beta, lambda_u, lambda_v, lambda_beta, lambda_w, r)

def test_data_sanity(X, Y, idx):

	print("X shape : ", X.shape,"\nY shape : ", Y.shape, "\n")
	feat_index = np.arange(X.shape[1])
	label_index = np.arange(Y.shape[1])
	print("Sparse representation of X[",idx,"] : \n", feat_index[X[idx]==1],"\n")
	print("Sparse representation of Y[", idx, "] : \n", label_index[Y[idx]==1],"\n")
	print("Number of ones in X[{}] : {} \n".format(idx, np.sum(X[idx])))
	print("Number of ones in Y[{}] : {} \n".format(idx, np.sum(Y[idx])))
# X, Y = read_datafile(args.dir_path, args.data_filename)
# test_data_sanity(X,Y, 1)
# X, Y = read_splitfile(args.dir_path, args.train_split_filename)
# test_data_sanity(X,Y, 1)
# X, Y = read_splitfile(args.dir_path, args.test_split_filename)
# test_data_sanity(X,Y, 1)

