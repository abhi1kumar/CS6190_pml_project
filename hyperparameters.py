import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_seen_labels = 80
num_iterations  = 1000

lambda_u    = 1.0
lambda_v    = 1.0
lambda_beta = 1.0
lambda_w    = 1.0
lambda_psi  = 1.0
r           = 5.0
K           = 128  # 80% of the total labels 500
cyclic      = True # Use cyclic loss while updating the weights
topk        = 1

print("========================================================================")
print("                         Parameters")
print("========================================================================")
print("Num Seen labels   = {}".format(num_seen_labels))
print("Num EM iterations = {}".format(num_iterations))
print("Latent factors K  = {}".format(K))
print("Cyclic Loss       = {}".format(cyclic))
print("Accuracy topk     = {}".format(topk))
print("")
