import argparse

from hyperparameters import *
from util import *
from test.test_read_data_split import *
from hyperparameters import *
from EM_Algorithm import *

parser = argparse.ArgumentParser(description='Read XML data from files')
parser.add_argument('--dir_path'            , type=str, default='data/Bibtex'        , help='path of the directory containing data-file')
parser.add_argument('--data_filename'       , type=str, default='Bibtex_data.txt'    , help='rel path of data_filename')
parser.add_argument('--train_split_filename', type=str, default='bibtex_trSplit.txt' , help='rel path of train_split_filename')
parser.add_argument('--test_split_filename' , type=str, default='bibtex_tstSplit.txt', help='rel path of test_split_filename')
args = parser.parse_args()

X, Y = get_data_labels_from_datafile(args.dir_path, args.data_filename)
X, Y = torch.tensor(X).float().to(device), torch.tensor(Y).float().to(device)

#normalise X
X = normaliseX(X)

#shuffle Y
num_all_labels  = Y.shape[1]
shuffled_indices = np.array(random.sample(range(num_all_labels), num_all_labels))
print("shuffled_indices : ", shuffled_indices)
Y = Y[:, shuffled_indices]

train_indices_for_each_split = get_indices_for_different_splits(args.dir_path, args.train_split_filename)
test_indices_for_each_split  = get_indices_for_different_splits(args.dir_path, args.test_split_filename)
num_splits = train_indices_for_each_split.shape[0]

# Get some stats
print("")
print("Number of splits             = {}".format(num_splits))
print("Train indices for each split =", train_indices_for_each_split.shape)
print("Test indices for each split  =", test_indices_for_each_split.shape)

iteration = 25

for split_idx in range(num_splits):
	train_indices = train_indices_for_each_split[split_idx]
	test_indices  = test_indices_for_each_split [split_idx]

	# Test data with seen and all labels
	test_X      = X[test_indices]
	test_Y_seen = Y[test_indices, :num_seen_labels]
	test_Y      = Y[test_indices]

	V    = torch.load(save_folder+"/v_"+str(iteration)+".pt")
	W    = torch.load(save_folder+"/w_"+str(iteration)+".pt")
	beta = torch.load(save_folder+"/beta_"+str(iteration)+".pt")
	psi  = get_psi(beta, V, lambda_psi)

	print("Test Data   X      shape =", test_X.shape)
	print("Test Labels Y seen shape =", test_Y_seen.shape)

	prec_seen = precision_at_k(test_X, test_Y, W, beta, psi, V, topk, num_seen_labels, isSeen = True)
	prec_seen_unseen = precision_at_k(test_X, test_Y, W, beta, psi, V, topk, num_seen_labels, isSeenUnseen = True)
	print("Test_Precision_Seen : @1 : {:.4f} \t @3 : {:.4f} \t @5 : {:.4f} \n Test_Precision_Seen+Unseen : @1 : {:.4f} \t @3  : {:.4f} \t @5 : {:.4f}".format
				(prec_seen[0], prec_seen[1], prec_seen[2], prec_seen_unseen[0], prec_seen_unseen[1], prec_seen_unseen[2]))
	print("\n========================================================================\n")
	break