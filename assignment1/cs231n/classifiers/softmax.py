import numpy as np
from random import shuffle

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

  num_classes = W.shape[1] #solution
  num_train = X.shape[0] #solution
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    Li = 0 # solution # The loss for i-th example.
    scores = X[i].dot(W) #solution # Compute the score for i-th example
    scores -= np.max(scores) #solution # This increases the numerical stability (https://cs231n.github.io/linear-classify/#loss) 
    correct_class_score = scores[y[i]] #solution # Find the correct class score.
    
    norm = np.sum(np.exp(scores))   #solution
    P = np.exp(scores)/norm #solution

    Li = -np.log(P[y[i]]) #solution

    for j in range(num_classes): #solution
      dW[:, j] += (P[j] -(j == y[i])) * X[i] #solution
    loss = loss + Li  #solution
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss/num_train #solution
  loss = loss + 0.5 * reg * np.sum(W * W) #solution

  dW = dW/num_train #solution
  dW = dW + 2*reg*W #solution


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1] #solution
  num_train = X.shape[0] #solution
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)  #solution
  scores -= np.max(scores) #solution
  norm = np.sum(np.exp(scores),axis=1).reshape(num_train,1) #solution
  P = np.exp(scores)/norm

  loss = np.sum(-np.log(P[(np.arange(num_train),y)]))/num_train #solution
  I = np.zeros_like(P) #solution
  I[np.arange(num_train),y] = 1 #solution
  dW = np.dot(X.T,P-I) #solution
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  loss = loss + 0.5*reg*np.sum(W*W) #solution

  dW = dW/num_train #solution
  dW = dW + 2*reg*W #solution


  return loss, dW

