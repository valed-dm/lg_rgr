import numpy as np
from scipy import sparse


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, x, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - x: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to 'x' for the bias sake.
        x = LogisticRegression.append_biases(x)
        num_train, dim = x.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            indices = np.random.choice(num_train, batch_size, replace=True)
            x_batch = x[indices]
            y_batch = y[indices]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad_w = self.loss(x_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w -= learning_rate * grad_w
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    def predict_proba(self, x, append_bias=False):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - x: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            x = LogisticRegression.append_biases(x)
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the probabilities of classes in y_proba.   #
        # Hint: It might be helpful to use np.vstack and np.sum                   #
        ###########################################################################
        # Compute the raw scores for the data points
        scores = x.dot(self.w)
        # Apply the sigmoid function to get probabilities for class 1
        prob_class_1 = 1 / (1 + np.exp(-scores))
        # Probability of class 0 is 1 minus the probability of class 1
        prob_class_0 = 1 - prob_class_1
        # Stack probabilities into an N x 2 array
        y_proba = np.vstack((prob_class_0, prob_class_1)).T
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_proba

    def predict(self, x):
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # Get probabilities for each class
        y_proba = self.predict_proba(x, append_bias=True)
        # Select the class with the higher probability
        y_pred = np.argmax(y_proba, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, x_batch, y_batch, reg):
        """Logistic Regression loss function
        Inputs:
        - x: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        loss = 0
        gradient = np.zeros_like(self.w)  # initialize the gradient as zero
        num_train = x_batch.shape[0]   # Number of training examples

        # Compute loss and gradient. Your code should not contain python loops.

        # Compute the scores (linear combination of inputs and weights)
        scores = x_batch.dot(self.w)
        # Apply the sigmoid function to get probabilities
        probs = 1 / (1 + np.exp(-scores))
        # Compute the difference between predicted probabilities and actual labels
        dscores = probs - y_batch

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead, so we divide by num_train.
        # Note that the same thing must be done with gradient.

        # Compute the average log loss over the batch
        loss = -np.mean(y_batch * np.log(probs) + (1 - y_batch) * np.log(1 - probs))
        gradient = x_batch.T.dot(dscores) / num_train

        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.

        # Add regularization to the loss (excluding the bias term)
        loss += 0.5 * reg * np.sum(self.w[:-1] ** 2)
        # Add regularization to the gradient (excluding the bias term)
        gradient[:-1] += reg * self.w[:-1]

        return loss, gradient

    @staticmethod
    def append_biases(x):
        return sparse.hstack((x, np.ones(x.shape[0])[:, np.newaxis])).tocsr()
