# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:36:43 2017

@author: Lucas Drumetz

ELMM: unmix a hyperspectral image
using the Extended Linear Mixing Model (ELMM). 

Details can be found in the paper:
    
L. Drumetz, M. Veganzones, S. Henrot, R. Phlypo, J. Chanussot and C. Jutten, 
"Blind Hyperspectral Unmixing Using an Extended Linear Mixing Model to Address
Spectral Variability," in IEEE Transactions on Image Processing, vol. 25, 
no. 8, pp. 3890-3905, Aug. 2016.
doi: 10.1109/TIP.2016.2579259

inputs: data: LxN data matrix (L: number of spectral bands, N: number of pixels)
        S0 : LxP reference endmember matrix (P: number of endmembers)
        A_init: PxN initial abundance matrix
        lambda_S : regularization parameter for the gaussian prior term
        
outputs: A: PxN abundance matrix
         psi: PxN scaling factor matrix
         S: LxPxN tensor containing the local endmember matrices
"""

import numpy as np
from proj_simplex import proj_simplex

def ELMM(data,A_init,S0,lambda_S):
    
    [L,N] = data.shape   
    
    [L,P] = S0.shape 
    
   # A = np.zeros([P,N])
    A = A_init
    S = np.zeros([L,P,N])
    psi = np.ones([P,N])
    
    for n in np.arange(N):
        S[:,:,n] = S0

    maxiter = 200
    
    U = A # split variable
    D = np.zeros(A.shape) # Lagrange mutlipliers

    rho = 1

    maxiter_ADMM = 100
    tol_A_ADMM = 10**-3
    tol_A = 10**-3
    tol_S = 10**-3
    tol_psi = 10**-3
    
    I = np.identity(P)

    for i in np.arange(maxiter):

        A_old = np.copy(A)
        psi_old = np.copy(psi)
        S_old = np.copy(S)

# A update

        for j in np.arange(maxiter_ADMM):
    
            A_old_ADMM = np.copy(A)
    
            for n in np.arange(N):
                A[:,n] = np.dot(np.linalg.inv(np.dot(np.transpose(S[:,:,n]),S[:,:,n]) + rho*I),np.dot(np.transpose(S[:,:,n]),data[:,n]) + rho*(U[:,n]-D[:,n]))
    
            U = proj_simplex(A+D)
    
            D = D + A - U
            
            if j > 0:
                rel_A_ADMM = np.abs((np.linalg.norm(A,'fro')-np.linalg.norm(A_old_ADMM,'fro')))/np.linalg.norm(A_old_ADMM,'fro')
  
                print("iteration ",j," of ",maxiter_ADMM,", rel_A_ADMM =",rel_A_ADMM)

                if rel_A_ADMM < tol_A_ADMM  :
                    break

# psi update

        for n in np.arange(N):
            for p in np.arange(P):
                psi[p,n] = np.dot(np.transpose(S0[:,p]),S[:,p,n])/np.dot(np.transpose(S0[:,p]),S0[:,p])

# S update


        for n in np.arange(N):
            S[:,:,n] = np.dot(np.outer(data[:,n],np.transpose(A[:,n])) + lambda_S*np.dot(S0,np.diag(psi[:,n])),np.linalg.inv(np.outer(A[:,n],np.transpose(A[:,n])) + lambda_S * I))
   
# termination checks    

        if i > 0:
    
            S_vec = np.hstack(S)

            rel_A = np.abs(np.linalg.norm(A,'fro')-np.linalg.norm(A_old,'fro'))/np.linalg.norm(A_old,'fro')
            rel_psi = np.abs(np.linalg.norm(psi,'fro')-np.linalg.norm(psi_old,'fro'))/np.linalg.norm(psi_old,'fro')
            rel_S = np.abs(np.linalg.norm(S_vec)-np.linalg.norm(np.hstack(S_old)))/np.linalg.norm(S_vec)
                
            print("iteration ",i," of ",maxiter,", rel_A =",rel_A, ", rel_psi =", rel_psi, "rel_S =", rel_S)                
                
            if rel_A < tol_A and rel_psi and tol_psi and rel_S < tol_S and i > 1:
                break
    
    return A,psi,S    
    