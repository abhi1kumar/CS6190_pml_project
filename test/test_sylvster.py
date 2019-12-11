
import os, sys
sys.path.append(os.getcwd())

import numpy as np
from lib.sylvester import *

def test(X, X_hat):
    print("Expected =", X)
    print("Obtained =", X_hat)
    diff = torch.max(torch.abs(X.reshape(-1) - X_hat.reshape(-1)))/X.numel()
    print("Max Diff =", diff)
    if diff < 1e-4:
        print("Test passed")
    else:
        print("Test failed!!!")
    print("")

A = torch.Tensor([[2.0]])
B = torch.Tensor([[4.0]])
C = torch.Tensor([[5.0]])

X_hat = solve_sylvester(A, B, C)
test(torch.Tensor([[0.8333333]]), X_hat)

A = torch.Tensor([[2.0, 0.0], [0, 1]])
B = torch.Tensor([[4.0, 0.0], [0, 1]])
C = torch.Tensor([[6.0, 1.0], [1, 6]])

X_hat = solve_sylvester(A, B, C)
test(torch.Tensor([[1, 0.333333], [0.2, 3]]), X_hat)

