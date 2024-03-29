

"""
	Sample Run:
	python train.py

	Start training the framework. 
 
"""
import argparse

from util import *
from EM_Algorithm import *
from test.test_read_data_split import *
from hyperparameters import *

parser = argparse.ArgumentParser(description='Read XML data from files')
parser.add_argument('--dir_path'            , type=str, default='data/Bibtex'        , help='path of the directory containing data-file')
parser.add_argument('--data_filename'       , type=str, default='Bibtex_data.txt'    , help='rel path of data_filename')
parser.add_argument('--train_split_filename', type=str, default='bibtex_trSplit.txt' , help='rel path of train_split_filename')
parser.add_argument('--test_split_filename' , type=str, default='bibtex_tstSplit.txt', help='rel path of test_split_filename')
args = parser.parse_args()

# Get the data and the labels on the dataset
X, Y = get_data_labels_from_datafile(args.dir_path, args.data_filename)
X, Y = torch.tensor(X).float().to(device), torch.tensor(Y).float().to(device)

#normalise X
X = normaliseX(X)

#shuffle Y
num_all_labels  = Y.shape[1]
#shuffled_indices = np.array(random.sample(range(num_all_labels), num_all_labels))
#print("shuffled_indices : ", shuffled_indices)
#Y = Y[:, shuffled_indices]

train_indices_for_each_split = get_indices_for_different_splits(args.dir_path, args.train_split_filename)
test_indices_for_each_split  = get_indices_for_different_splits(args.dir_path, args.test_split_filename)
num_splits = train_indices_for_each_split.shape[0]

# Get some stats
print("")
print("Number of splits             = {}".format(num_splits))
print("Train indices for each split =", train_indices_for_each_split.shape)
print("Test indices for each split  =", test_indices_for_each_split.shape)

for split_idx in range(num_splits):
	train_indices = train_indices_for_each_split[split_idx]
	test_indices  = test_indices_for_each_split [split_idx]

	# Train data with seen and all labels
	train_X      = X[train_indices];
	train_Y_seen = Y[train_indices, :num_seen_labels];
	#num_unseen_labels = num_all_labels - num_seen_labels
	#train_Y_seen = Y[train_indices, num_unseen_labels:]
	train_Y      = Y[train_indices]

	# Test data with seen and all labels
	test_X      = X[test_indices]
	test_Y_seen = Y[test_indices, :num_seen_labels]
	#test_Y_seen = Y[test_indices, num_unseen_labels:]
	test_Y      = Y[test_indices]

	# Label co-occurrence matrix
	M = torch.matmul(train_Y.t(), train_Y)[:num_seen_labels,:].float().to(device) 
	#M  = torch.matmul(train_Y.t(), train_Y)[num_unseen_labels:, :].float().to(device)
	# Initialisation
	if(init_method == "xavier_init"):
		V    = torch.normal(mean = 0, std = np.sqrt(1/train_Y_seen.shape[1]), size = (train_Y_seen.shape[1], K)).float().to(device)
		W    = torch.normal(mean = 0, std = np.sqrt(1/train_X.shape[1]), size = (train_X.shape[1], K)).float().to(device)
		beta = torch.normal(mean = 0, std = np.sqrt(1/M.shape[1]), size = (M.shape[1], K)).float().to(device)
		print("Xavier initialisation done !!!")
	else : 
		V    = torch.normal(mean = 0, std = 1, size = (train_Y_seen.shape[1], K)).float().to(device)
		W    = torch.normal(mean = 0, std = 1, size = (train_X.shape[1], K)).float().to(device)
		beta = torch.normal(mean = 0, std = 1, size = (M.shape[1], K)).float().to(device);


	U = torch.matmul(train_X, W)
	print("Train Data   X      shape =", train_X.shape)
	print("Train Labels Y seen shape =", train_Y_seen.shape)

	print("\n========================================================================")
	print("                             EM Algorithm")
	print("========================================================================\n")

	for i in range(1+num_iterations):
		U, V, beta, W, psi = EM_algorithm(train_X, train_Y_seen, V, U, M, W, beta, lambda_u, lambda_v, lambda_beta, lambda_w, lambda_psi, r, cyclic)
		prec_train = precision_at_k(train_X, train_Y_seen, W, beta, psi, V, topk, num_seen_labels, isSeen = True)
		prec_test  = precision_at_k(test_X, test_Y_seen, W, beta, psi, V, topk, num_seen_labels, isSeen = True)
		print("Iteration : {:3d} \t Train_Precision_Seen : @1 : {:.4f} \t @3 : {:.4f} \t @5 : {:.4f} \t Test_Precision_Seen : @1 : {:.4f} \t @3  : {:.4f} \t @5 : {:.4f}".format
				(i, prec_train[0], prec_train[1], prec_train[2], prec_test[0], prec_test[1], prec_test[2]))
		if(i==0 or i%save_iter == 0 ):

			torch.save(V, save_folder + "/v_"+str(i)+".pt")
			torch.save(beta, save_folder + "/beta_"+str(i)+".pt")
			torch.save(W, save_folder + "/w_"+str(i)+".pt")
			torch.save(psi, save_folder + "/psi_"+str(i))
			#prec_train = precision_at_k(train_X, train_Y, W, beta, psi, V, topk, num_seen_labels, isUnseen = True)
			#prec_test  = precision_at_k(test_X, test_Y, W, beta, psi, V, topk, num_seen_labels, isUnseen = True)
			print("Iteration : {:3d} \t Train_Precision_Unseen : @1 : {:.4f} \t @3 : {:.4f} \t @5 : {:.4f} \t Test_Precision_Unseen : @1 : {:.4f} \t @3  : {:.4f} \t @5 : {:.4f}".format
					(i, prec_train[0], prec_train[1], prec_train[2], prec_test[0], prec_test[1], prec_test[2]))
		
	print("                           EM Algorithm Completed")
	print("\n========================================================================\n")
	break
