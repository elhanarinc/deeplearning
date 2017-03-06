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

    total_f_i = np.sum(np.exp(f_i))

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

  # Subtract the max value
  f -= np.max(f)

  # -f[y[i]] values
  correct_f_values = f[y, np.arange(num_train)]

  # total_f values
  total_f_values = np.sum(np.exp(f), axis = 0)

  # -f[y] + log(total_f)
  loss = np.sum(np.log(total_f_values) - correct_f_values)

  # Exponential of f values
  exp_f_values = np.exp(f)

  # Probabilities
  p = exp_f_values / total_f_values

  # Subtract the correct class values
  correct_classes = np.zeros(shape = (num_classes, num_train))
  correct_classes[y, np.arange(num_train)] = 1
  dW = p.dot(X.T) - correct_classes.dot(X.T)

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
