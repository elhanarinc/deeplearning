import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  num_classes = W.shape[0]
  num_train = X.shape[1]

  for i in xrange(num_train):

    # shape 3073x1
    X_at_i = X[:, i]

    # f_i value
    f_i = W.dot(X_at_i)

    # Set log C and apply in order to block inconsistency
    f_i -= np.max(f_i)

    total_f_i = 0

    # Find the sum of e score values
    for my_iter in f_i:
      total_f_i += np.exp(my_iter)

    # Formula from the class
    loss += -f_i[y[i]] + np.log(total_f_i)

    for my_iter in range(num_classes):
      # Find the probability of one class
      p = np.exp(f_i[my_iter]) / total_f_i

      # Gradient formula is "(pk - 1{yi = k}).X"
      if my_iter == y[i]:
        dW[my_iter] += (p - 1) * X_at_i
      else:
        dW[my_iter] += p * X_at_i

  loss /= num_train

  # Calculate average
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[0]
  num_train = X.shape[1]

  # Find f values; shape 10x18000
  f = W.dot(X)

  # Find the maximum f and subtract it from every class value
  f -= np.max(f, axis = 0, keepdims = True)

  # Find the total_f for each class; shape 1x18000
  total_f = np.sum(np.exp(f), axis = 0, keepdims = True)

  # Find the probabilities; shape 10x18000
  p = np.exp(f)/total_f

  loss = np.sum(-np.log(p[y, np.arange(num_train)]))

  correct_classes = np.zeros_like(p)
  correct_classes[y, np.arange(num_train)] = 1
  dW = (p - correct_classes).dot(X.T)

  loss /= num_train

  # Calculate average
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
