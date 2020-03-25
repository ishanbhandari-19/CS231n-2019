from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_example = X.shape[0]
    num_classes = W.shape[1]
    scores = np.zeros((num_example,num_classes))
    for i in range(num_example):
        scores[i,:] = X[i]@W
        exp = np.exp(scores[i,:])
        correct_class = exp[y[i]]
        sum_ = np.sum(exp)
        loss -=  np.log(correct_class/sum_)
        for j in range(num_classes):
            if j==y[i]:
                dW[:,j] -= X[i]*(1-(correct_class/sum_))
            else:
                dW[:,j] += X[i]*(exp[j]/sum_)
            
    
    loss/=num_example
    loss = loss + 0.5*reg*np.sum(W*W)
    dW /=num_example
    dW += reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    scores = X@W
    exp = np.exp(scores)
    correct_class = exp[np.arange(num_train),y].reshape((num_train,1))
    scores_sum = np.sum(exp,axis=1).reshape((num_train,1))
    loss = np.sum(-1*np.log(correct_class/scores_sum))
    loss/=num_train
    loss += 0.5*reg*np.sum(W*W)
    exp = exp/scores_sum
    exp[np.arange(num_train),y] -=1
    dW = X.T@exp
    dW/=num_train
    dW += reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
