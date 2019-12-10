import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_iterations  = 100
num_seen_labels = 110

lambda_u    = 1.0
lambda_v    = 1.0
lambda_beta = 1.0
lambda_w    = 1.0
lambda_psi  = 1.0
r           = 5.0
K           = 500
topk        = 5