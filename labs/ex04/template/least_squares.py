# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

from costs import *

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    """calculate the least squares solution."""
    # a = tx.T.dot(tx)
    # b = tx.T.dot(y)
    # w = np.linalg.solve(a, b)
    # return w, compute_loss(y, tx, w)

    # The lest squares solution of a linear system Ax = b is A.T A x = A.T B <=> b = (A.T)^-1 A.T A x
    # tx.T @ tx @ w = tx.T @ y
    w = np.linalg.lstsq(tx, y, rcond=None)[0]
    # w_ = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


least_squares(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
