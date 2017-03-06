import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    # shape 3073x1
    X_at_i = X[:, i]

    # shape 10x1
    scores = W.dot(X_at_i)

    for j in xrange(num_classes):
      # Skip and ignore the same class value
      if j == y[i]:
        continue

      # Get the margin score
      margin = scores[j] - scores[y[i]] + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # Increase the correct one's gradient, since there is margin clearly y[i] is not the correct class
        dW[j] += X_at_i.T
        # Decrease the incorrect one's gradient
        dW[y[i]] -= X_at_i.T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Calculate average
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[0]
  num_train = X.shape[1]

  scores = W.dot(X)

  # Get the correct class scores
  correct_class_score = scores[y, np.arange(num_train)]

  # Find margins; 10x18000 - 18000 (subtract correct class scores from each classes)
  margins = np.maximum(0, scores - correct_class_score + 1)

  # Do not include the same class value
  margins[y, np.arange(num_train)] = 0

  # Sum up all margins
  loss = np.sum(margins)

  loss /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # Binarize the values of margin matrix, due to see the total num of classes which is
  # greater than 0
  margins[margins > 0] = 1

  margins[y, np.arange(num_train)] = -np.sum(margins, axis = 0)
  dW = margins.dot(X.T)

  # Calculate average
  dW /= num_train

  # Add regularization
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
