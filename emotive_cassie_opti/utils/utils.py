#!/usr/bin/python
import numpy as np
import math
from six.moves import cPickle as pickle 
from scipy.special import comb

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def dbezier(coeff, s):
    dcoeff = __diff_coeff(coeff)
    fcn = bezier(dcoeff,s)
    return fcn

def __binomial(i, n):
    """Binomial coefficient"""
    return math.factorial(n) / float(
        math.factorial(i) * math.factorial(n - i))

def __bernstein(t, i, n):
    """Bernstein polynom"""
    return __binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))

def bezier(coeff, s):
    """Calculate coordinate of a point in the bezier curve"""
    n, m = coeff.shape[0], coeff.shape[1]
    m = m - 1
    fcn = np.zeros((n, 1))
    for k in range(m+1):
        fcn += coeff[:,k].reshape((n,1)) * __bernstein(s, k, m)
    return fcn.reshape((n,))

def __diff_coeff(coeff):
    M = coeff.shape[1] - 1
    A = np.zeros((M, M+1))

    for i in range(M):
        A[i,i] = -(M-i)*comb(M,i)/comb(M-1, i)
        A[i,i+1] = (i+1)*comb(M,i+1)/comb(M-1,i)
    
    A[M-1,M] = M*comb(M,M)
    # dcoeff = coeff@(A.T)
    dcoeff = np.matmul(coeff, A.T)    
    return dcoeff