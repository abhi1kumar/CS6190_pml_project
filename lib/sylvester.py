

"""
    Solves the sylvester matrix equation
    AX + XB = C
    where A, B and C are known and X is to be determined

    Reference
    Solution of the Equation AX + XB = C by Inversion of an MxM or N×N matrix
    Antony Jameson, SIAM, 1968
    aero-comlab.stanford.edu/Papers/jameson_007.pdf
"""

import torch
TOLERANCE = 1e-5

def solve_sylvester(A, B, C):
    """
        Solves AX + XB = C where A, B and C are all tensors
        A = shape D x D
        B = shape K x K
        C = shape D x K

        Assumptions:
        A, B and C should be 2D matrices.

        A and B should be symmetric positive semi-definite matrices. 

        This is needed since we are using torch.symeig function.
        Only this function in Pytorch supports backward pass
        https://pytorch.org/docs/stable/torch.html#torch.symeig
    
        Returns tensor
        output = shape D x K    
    """
 
    eig_val_A, eig_vec_A = torch.symeig(A, eigenvectors= True)
    eig_val_B, eig_vec_B = torch.symeig(B, eigenvectors= True)

    dim_A = A.shape[0]
    dim_B = B.shape[0]

    # Get C_hat
    C_hat = torch.matmul(torch.matmul(eig_vec_A.t(), C), eig_vec_B)

    # Now get another tensor with same size as C_hat
    # Make \mu_i + \nu_j
    X_hat_deno = eig_val_A.view(-1, 1).repeat(1, dim_B) + eig_val_B.view(1, -1).repeat(dim_A, 1)

    X_hat = C_hat * 1.0/(X_hat_deno + TOLERANCE)

    output= torch.matmul(torch.matmul(eig_vec_A.t(), X_hat), eig_vec_B.t())
    
    return output
