

"""
	Sample Run:
	python3 train.py

	Trains Probabilistic Framework mentioned in 
	A Probabilistic Framework for Zero-Shot Multi-Label Learning
	A Gaure, A Gupta, V Verma, P Rai, UAI 2017
	http://auai.org/uai2017/proceedings/papers/278.pdf

	Version 1 2019-10-15
"""
import torch
import numpy as np
import torch.nn.functional as F

# ============================================================
# Global variables
# ============================================================
lambda_w  = 1.0
lambda_si = 1.0




def E_step(U, V, Beta, r=0):
	"""
		Calculates the E step of the algorithm
		
		Input:
		U    = shape N  x K
		V    = shape Ls x K
		Beta = shape L  x K

		Output:
		omega = shape N  x Ls
		tau   = shape Ls x L
	"""
	zheta = torch.matmul(U, torch.transpose(V, 0, 1)) # N x Ls
	omega = 1/(2*zheta) * F.tanh(zheta/2)             # N x Ls
	
	gamma = torch.matmul(V, torch.transpose(Beta, 0, 1)) # Ls x L
	M = torch.zeros(gamma.shape)
	tau = (M +r)/ (2*gamma) * F.tanh(gamma/2)

	return omega, tau

def ridge_solver_batch(Sigma, Vector):
	"""
		Solves ridge regression batch wise
		
		Inputs:
		Sigma = shape N x K x K
		Vector = shape N x K

		Output:
		output = shape N x K
	"""
	#output = torch.zeros((Vector.shape[0], Sigma.shape[1]))

	# Add a new dimension so that matrix multiplication could be applied
	Vector = Vector.view[:,:, :]

	# Inverse works in batches
	# https://pytorch.org/docs/stable/torch.html#torch.inverse
	Sigma_inverse = torch.inverse(Sigma)
	output = torch.bmm(Sigma_inverse, Vector)
	
	return output

def M_step(X, U, V, Beta):
	"""
		Calculates the M step of the EM algorithm

		Inputs:
		X    = shape N  x D
		V    = shape Ls x K
		Beta = shape L  x K

		Output:
		W    = shape D x K
	"""
	# Five update equations

	# Update for W
	Sigma_w = torch.matmul(torch.transpose(X, 0, 1), X) + lambda_w * torch.eye(X.shape[1]).type(X.type())
	W_t = torch.inverse(Sigma_w) * torch.transpose(X, 0, 1) * U
	W = W.transpose(0, 1)