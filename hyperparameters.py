import torch
import numpy as np
import random

seed_val = 11
random.seed(seed_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_seen_labels = 80
num_iterations  = 500

lambda_u    = 1.0
lambda_v    = 1.0
lambda_beta = 1.0
lambda_w    = 1.0
lambda_psi  = 1.0
r           = 5
K           = 128  # 80% of the total labels 159
cyclic      = False # Use cyclic loss while updating the weights
topk        = 5
init_method = "xavier_init"	#Use xavier_init for xavier initialisation

print("========================================================================")
print("                         Parameters")
print("========================================================================")
print("Num Seen labels   = {}".format(num_seen_labels))
print("Num EM iterations = {}".format(num_iterations))
print("Latent factor K   = {}".format(K))
print("Parameter r       = {}".format(r))
print("init_method       = {}".format(init_method))
print("Cyclic Loss       = {}".format(cyclic))
print("Accuracy topk     = {}".format(topk))
print("")

save_iter   = 5
save_folder = "run_1"
print("save_folder       = {}".format(save_folder))
