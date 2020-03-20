# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:36:43 2017

@author: Lucas Drumetz

RELMM: unmix a hyperspectral image
using the Robust version of the Extended Linear Mixing Model (ELMM). 

Details can be found in the paper

L. Drumetz, J. Chanussot, C. Jutten, W. Ma and A. Iwasaki, "Spectral 
Variability Aware Blind Hyperspectral Image Unmixing Based on Convex Geometry,"
in IEEE Transactions on Image Processing, vol. 29, pp. 4568-4582, 2020.
doi: 10.1109/TIP.2020.2974062

inputs: data: LxN data matrix (L: number of spectral bands, N: number of 
                               pixels)
        S0 : LxP reference endmember matrix (P: number of endmembers)
        A_init: PxN initial abundance matrix
        lambda_S : regularization parameter for the gaussian prior term
        lambda_S0 : regularization parameter for the volume term
        
outputs: A: PxN abundance matrix
         psi: PxN scaling factor matrix
         S: LxPxN tensor containing the local endmember matrices
         S0: LxP updated reference endmember matrix
         
NB: This code requires the pymanopt package to be installed. It allows to use 
various optimization algorithms on Riemannian Manifolds. Here we use it to 
constrain the solution of the optimization problem on S0 to lie on the unit 
sphere.
"""

import numpy as np
from proj_simplex import proj_simplex
from pymanopt.manifolds import Oblique
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient

def RELMM(data,A_init,S0,lambda_S,lambda_S0):   
    
    [L,N] = data.shape   
    
    [L,P] = S0.shape 
    
    V = P*np.eye(P) - np.outer(np.ones(P),np.transpose(np.ones(P)))    
    
    
    def cost(X):

        data_fit = np.zeros(N)    
    
        for n in np.arange(N):
            data_fit[n] = np.linalg.norm(S[:,:,n]-np.dot(X,np.diag(psi[:,n])),'fro')**2 

        cost = lambda_S/2 * np.sum(data_fit,axis = 0) + lambda_S0/2 * np.trace(np.dot(np.dot(X,V),np.transpose(X)))

        return cost        
        
    def egrad(X):

        partial_grad = np.zeros([L,P,N])    
    
        for n in np.arange(N):
            partial_grad[:,:,n] = np.dot(X,np.diag(psi[:,n]))-np.dot(S[:,:,n],np.diag(psi[:,n]))

        egrad = lambda_S * np.sum(partial_grad,axis = 2) + lambda_S0*np.dot(X,V)
    
        return egrad   
    

    
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
    tol_S0 = 10**-3
    
    I = np.identity(P)

    for i in np.arange(maxiter):

        A_old = np.copy(A)
        psi_old = np.copy(psi)
        S_old = np.copy(S)
        S0_old = np.copy(S0)

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
   
# S0 update

        manifold = Oblique(L, P)
        solver = ConjugateGradient()
        problem = Problem(manifold=manifold, cost=cost, egrad = egrad)
        S0 = solver.solve(problem)   
   
# termination checks    

        if i > 0:
    
            S_vec = np.hstack(S)

            rel_A = np.abs(np.linalg.norm(A,'fro')-np.linalg.norm(A_old,'fro'))/np.linalg.norm(A_old,'fro')
            rel_psi = np.abs(np.linalg.norm(psi,'fro')-np.linalg.norm(psi_old,'fro'))/np.linalg.norm(psi_old,'fro')
            rel_S = np.abs(np.linalg.norm(S_vec)-np.linalg.norm(np.hstack(S_old)))/np.linalg.norm(S_old)
            rel_S0 = np.abs(np.linalg.norm(S0,'fro')-np.linalg.norm(S0_old,'fro'))/np.linalg.norm(S0_old,'fro')                 
                
            print("iteration ",i," of ",maxiter,", rel_A =",rel_A, ", rel_psi =", rel_psi, "rel_S =", rel_S, "rel_S0 =", rel_S0)                
                
            if rel_A < tol_A and rel_psi and tol_psi and rel_S < tol_S  and rel_S0 < tol_S0 and i > 1:
                break
    
    

    
    return A,psi,S,S0    
    