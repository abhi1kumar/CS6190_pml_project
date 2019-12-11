

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
from util import *
from lib.sylvester import *

# ============================================================
# Global variables
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    Vector = Vector[:, :, None]

    # Inverse works in batches
    # https://pytorch.org/docs/stable/torch.html#torch.inverse
    Sigma_inverse = torch.inverse(Sigma)
    output = torch.bmm(Sigma_inverse, Vector).squeeze()
    
    return output

def get_psi(beta, V, Ls, lambda_psi):
    """
        Calculates psi once beta and V have been computed

        Inputs:
        beta = shape L  x K
        V    = shape Ls  x K

        Output:
        psi    = shape K x K
    """

    # Compute psi
    beta_ls = beta[:Ls, :]                                      #beta_ls   = shape Ls x K
    beta1   = beta_ls[:, :, None]                               #beta1     = shape Ls x K x 1
    beta2   = beta_ls[:, None, :]                               #beta2     = shape Ls x 1 x K
    betabetat = torch.sum(torch.matmul(beta1, beta2), axis=0)   #betabetat = shape K  x K   
    psi = torch.matmul((betabetat + lambda_psi*torch.eye(V.shape[1]).to(device)).inverse(),torch.matmul(beta_ls.t(), V))   #psi = shape K x K

    return psi

def update_W(X, U, lambda_w, cyclic):
    """
        Calculates the updated U for the M step of the EM algorithm

        Inputs:
        X    = shape N  x D
        U    = shape N  x K
        Y    = shape N  x Ls
        V    = shape Ls x K

        Output:
        W    = shape D x K
    """
    if cyclic:
        # Solve Sylvster equation
        Z = torch.matmul(Y, V)  # N x K
        A = torch.matmul(torch.transpose(X, 0, 1), X)  # D x D
        B = torch.matmul(torch.transpose(Z, 0, 1), Z)  # K x K
        C = torch.matmul(torch.transpose(X, 0, 1), U) + torch.matmul(torch.transpose(X, 0, 1), Z) # D x K

        # Solve sylvester equations
        # AW + WB = C
        W = solve_sylvester(A, B, C)
    else:
        sigma_w = torch.matmul(torch.transpose(X, 0, 1), X) + lambda_w * torch.eye(X.shape[1]).type(X.type())
        W = torch.matmul(torch.inverse(sigma_w), torch.matmul(torch.transpose(X, 0, 1), U))
 
    return W

def update_U(X, Y, V, W, omega, lambda_u):
    """
        Calculates the updated U for the M step of the EM algorithm

        Inputs:
        X    = shape N  x D
        Y    = shape N  x Ls
        V    = shape Ls x K
        W    = shape D x K 
        omega= shape N x Ls

        Output:
        U    = shape N x K
    """

    # Update for U

    kappa   = (Y - 0.5)[:,:,None].repeat(1,1,V.shape[1])        #kappa = shape N x Ls x K
    sigma_u = torch.empty(X.shape[0], V.shape[1], V.shape[1]).to(device)   #sigma_U = shape N x K x K
    v1      = V[:, :, None]             #v1 = shape Ls x K x 1
    v2      = V[:, None, :]             #v2 = shape Ls x 1 x K
    vvt     = torch.matmul(v1,v2)       #v2 = shape Ls x K x K

    for i in range(X.shape[0]):
        omega_i    = (omega[i])[:, None, None].repeat(1, V.shape[1], V.shape[1])        #omega_i = shape Ls x K x K
        sigma_u[i] = (torch.sum(omega_i*vvt, axis=0) + lambda_u*torch.eye(V.shape[1]).to(device)).inverse()   #sigma_u[i] = shape K x K

    term1 = torch.sum(kappa*V, axis=1).squeeze()    #term1 = shape N x K
    term2 = lambda_u*torch.matmul(X,W)              #term2 = shape N x K
    U     = torch.matmul(sigma_u, (term1 + term2)[:,:,None]).squeeze()
    
    return U

def update_V(Y, M, U, beta, omega, tau, lambda_v, r):
    """
        Calculates the updated U for the M step of the EM algorithm

        Inputs:
        Y    = shape N  x Ls
        M    = shape Ls x L
        U    = shape N  x K
        beta = shape L  x K
        omega= shape N  x Ls
        tau  = shape Ls x L

        Output:
        V    = shape Ls x K
    """
    kappa_y = ((Y - 0.5).t())[:,:,None].repeat(1,1,U.shape[1])      #kappa = shape Ls x N x K
    kappa_m = 0.5*(M + r)[:,:,None].repeat(1,1,beta.shape[1])               #kappa_m = shape Ls x L x K
    sigma_V = torch.empty(Y.shape[1], U.shape[1], U.shape[1]).to(device)              #sigma_V = shape Ls x K x K
    u1      = U[:, :, None]             #u1 = shape N x K x 1
    u2      = U[:, None, :]             #u2 = shape N x 1 x K
    uut     = torch.matmul(u1,u2)       #uut = shape N x K x K
    omega_t = omega.t()                 #omega_t = shape Ls x N
    beta1   = beta[:, :, None]          #beta1   = shape L  x K x 1
    beta2   = beta[:, None, :]          #beta2   = shape L  x 1 x K
    betabetat = torch.matmul(beta1, beta2)  #betabetat = shape L x K x K
    for i in range(Y.shape[1]):
        omega_t_i    = (omega_t[i])[:, None, None].repeat(1, U.shape[1], U.shape[1])        #omega_i = shape N x K x K
        tau_i      = (tau[i])[:, None, None].repeat(1, beta.shape[1], beta.shape[1])        #tau_i = shape L x K x K
        sigma_V[i] = (torch.sum(omega_t_i*uut, axis=0) + torch.sum(tau_i*betabetat, axis=0) + lambda_v*torch.eye(U.shape[1]).to(device)).inverse()   #sigma_u[i] = shape K x K

    term1 = torch.sum(kappa_y*U, axis=1).squeeze()                  #term1 = shape Ls x K
    term2 = torch.sum(kappa_m*beta, axis=1).squeeze()               #term2 = shape Ls x K
    V     = torch.matmul(sigma_V, (term1 + term2)[:,:,None]).squeeze()
    
    return V

def update_beta(M, V, tau, lambda_beta, r):
    """
        Calculates the updated U for the M step of the EM algorithm

        Inputs:
        M    = shape Ls x L
        V    = shape Ls x K
        tau  = shape Ls x L

        Output:
        beta = shape L  x K
    """

    # Update for beta

    kappa_m    = (0.5*(M + r).t())[:,:,None].repeat(1,1,V.shape[1])  #kappa_m = shape L x Ls x K
    sigma_beta = torch.empty(M.shape[1], V.shape[1], V.shape[1]).to(device)        #sigma_beta = shape L x K x K
    v1      = V[:, :, None]             #v1 = shape Ls x K x 1
    v2      = V[:, None, :]             #v2 = shape Ls x 1 x K
    vvt     = torch.matmul(v1,v2)       #v2 = shape Ls x K x K
    tau_t   = tau.t()                   #tau_t = shape L x Ls
    for i in range(M.shape[1]):
        tau_t_i    = (tau_t[i])[:, None, None].repeat(1, V.shape[1], V.shape[1])        #tau_t_i = shape Ls x K x K
        sigma_beta[i] = (torch.sum(tau_t_i*vvt, axis=0) + lambda_beta*torch.eye(V.shape[1]).to(device)).inverse()   #sigma_u[i] = shape K x K

    term1 = torch.sum(kappa_m*V, axis=1).squeeze()              #term1 = shape L x K
    beta  = torch.matmul(sigma_beta, term1 [:,:,None]).squeeze()#beta  = shape L x K
    
    return beta

def E_step(U, V, beta, M, r):
    """
        Calculates the E step of the algorithm
        
        Input:
        U    = shape N  x K
        V    = shape Ls x K
        beta = shape L  x K

        Output:
        omega = shape N  x Ls
        tau   = shape Ls x L
    """

    zheta = torch.matmul(U, torch.transpose(V, 0, 1)) # N x Ls
    omega = (1/(2*zheta)) * torch.tanh(zheta/2)             # N x Ls
    
    gamma = torch.matmul(V, torch.transpose(beta, 0, 1)) # Ls x L
    tau = ((M +r)/ (2*gamma)) * torch.tanh(gamma/2) # Ls x L

    return omega, tau

def M_step(X, Y, V, U, M, W, beta, tau, omega, lambda_u, lambda_v, lambda_beta, lambda_w, r, cyclic):
    """
        Calculates the M step of the EM algorithm

        Inputs:
        X    = shape N  x D
        V    = shape Ls x K
        U    = shape N  x K
        M    = shape Ls x L
        W    = shape D  x K
        beta = shape L  x K
        tau  = shape Ls x L
        omega= shape N  x Ls

        Outputs: 
        V    = shape Ls x K
        U    = shape N  x K
        W    = shape D  x K
        beta = shape L  x K
    """

    # Update U, V, beta, W
    U    = update_U(X, Y, V, W, omega, lambda_u)
    #print("U updated")
    V    = update_V(Y, M, U, beta, omega, tau, lambda_v, r)
    #print("V updated")
    beta = update_beta(M, V, tau, lambda_beta, r)
    #print("beta updated")
    W    = update_W(X, U, lambda_w, cyclic)
    #print("W updated")

    return U, V, beta, W

def EM_algorithm(iterations, X, Y, Y_all, V, U, M, W, beta, lambda_u, lambda_v, lambda_beta, lambda_w, lambda_psi, r, Ls, test_X, test_Y, topk, cyclic):
    """
        Calculates the M step of the EM algorithm

        Inputs:
        X    = shape N  x D
        V    = shape Ls x K
        U    = shape N  x K
        M    = shape Ls x L
        W    = shape D  x K
        beta = shape L  x K
        cyclic = True # Suggests to use cyclic loss while training

    """

    # EM algorithm
    print("\n==================================================================================================\n")
    print("                                         EM Algorithm")
    print("\n==================================================================================================\n")

    for i  in range(iterations):
        omega, tau    = E_step(U, V, beta, M, r)
        #print("E-Step Done".format(i))
        U, V, beta, W = M_step(X, Y, V, U, M, W, beta, tau, omega, lambda_u, lambda_v, lambda_beta, lambda_w, r, cyclic)
        #print("M-Step Done".format(i))
        psi = get_psi(beta, V, Ls, lambda_psi)
        precision_train = precision_at_k(X, Y_all, W, beta, psi, topk)
        precision_test = precision_at_k(test_X, test_Y, W, beta, psi, topk)
        print("Iter : {:3d} \t Precision Train : {:.4f} \t Precision Test : {:.4f}".format(i, precision_train, precision_test))

    psi = get_psi(beta, V, Ls, lambda_psi)
    print("psi calculated")
    return U, V, beta, W, psi
