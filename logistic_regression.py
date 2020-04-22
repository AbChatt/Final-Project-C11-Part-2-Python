"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Exam
B. Chan, D. Fleet

===========================================================
 COMPLETE THIS TEXT BOX:

 Student Name: Abhishek Chatterjee
 Student number: 1004820615
 UtorID: chatt114

 I hereby certify that the work contained here is my own


 _Abhishek Chatterjee_
 (sign with your name)
===========================================================
"""

import numpy as np

from utils import softmax

class LogisticRegression:
    def __init__(self,
                 num_features,
                 num_classes,
                 rng=np.random):
        """ This class represents a multi-class logistic regression model.
        NOTE: We assume the labels are 0 to K-1, where K is number of classes.

        self.parameters contains the vector of model weights.
        NOTE: the bias term is assume to be the B first element of the vecto

        TODO: You will need to implement one method of this class:
        - _compute_loss_and_gradient: ndarray, ndarray -> float, ndarray

        Implementation description will be provided under each method.
        
        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - K: Number of classes.

        Args:
        - num_features (int): The dimension of feature vectors for input data.
        - num_classes (int): The number of classes in the task.
        - rng (RandomState): The random number generator to initialize weights.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.rng = rng

        # Initialize parameters
        self.parameters = np.zeros(shape=(num_classes, self.num_features + 1))

    def init_weights(self, factor=1, bias=0):
        """ This initializes the model weights with random values.

        Args:
        - factor (float): A constant scale factor for the initial weights.
        - bias (float): The bias value
        """
        self.parameters[:, 1:] = factor * self.rng.rand(self.num_classes, self.num_features)
        self.parameters[:, 0] = bias

    def _compute_loss_and_gradient(self, X, y):
        """ This computes the training loss and its gradient. 
 	    That is, the negative log likelihood (NLL) and the gradient of NLL.

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix containing N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector containing N scalar outputs (labels).

        Output:
        - nll (float): The NLL of the training inputs and outputs.
        - grad (ndarray (shape: (K, D + 1))): A Kx(D+1) weight matrix (including bias) containing the gradient of NLL
                                              (i.e. partial derivatives of NLL w.r.t. self.parameters).
        """
        (N, D) = X.shape
        # ====================================================
        # TODO: Implement your solution within the box

        alpha_inv = 0
        beta_inv = 0    # change these values to formulas
        
        X = np.hstack((np.ones(shape = (np.shape(X)[0], 1), dtype = float), X))

        C_inv = np.zeroes((np.shape(self.parameters)[0], np.shape(self.parameters)[0]), dtype = float)

        C_inv[0][0] = beta_inv

        for i in range(1, np.shape(self.parameters)[0]):
            C_inv[i][i] = alpha_inv
        
        parameters_transpose = np.transpose(self.parameters)
        nll = 0.5 * (parameters_transpose @ C_inv @ self.parameters)

        for j in range(np.shape(X)[0]):
            single_x = X[j, ].reshape(1, np.shape(X)[1])
            nll  = nll - (y[j] * log_sigmoid_class_1(np.matmul(single_x, self.parameters)) + (1 - y[j]) * log_sigmoid_class_0(np.matmul(single_x @ self.parameters)))
        
        grad = np.matmul(C_inv, self.parameters)

        for k in range(np.shape(X)[0]):
            single_x = X[k, ].reshape(1, np.shape(X)[1])
            grad = grad + (sigmoid(np.matmul(single_x @ self.parameters)) - y[k]) * np.transpose(single_x)
        
        nll = nll[0][0]

        # ====================================================

        return nll, grad

    def learn(self,
              train_X,
              train_y,
              num_epochs=1000,
              step_size=1e-3,
              check_grad=False,
              verbose=False,
              eps=np.finfo(np.float).eps):
        """ This performs gradient descent to find the optimal model parameters given the training data.

        NOTE: This method mutates self.parameters

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix containing N D-dimensional training inputs.
        - train_y (ndarray (shape: (N, 1))): A N-column vector containing N scalar training outputs (labels).
        - num_epochs (int): Number of gradient descent steps
                        NOTE: 1 <= num_epochs
        - step_size (float): Gradient descent step size
        - check_grad (bool): Whether or not to check gradient using finite difference.
        - verbose (bool): Whether or not to print gradient information for every step.
        - eps (float): Machine epsilon

        ASIDE: The design for applying gradient descent to find local minimum is usually different from this.
               You should think about a better way to do this! Scipy is a good reference for such design.
        """
        assert len(train_X.shape) == len(train_y.shape) == 2, f"Input/output pairs must be 2D-arrays. train_X: {train_X.shape}, train_y: {train_y.shape}"
        (N, D) = train_X.shape
        assert N == train_y.shape[0], f"Number of samples must match for input/output pairs. train_X: {N}, train_y: {train_y.shape[0]}"
        assert D == self.num_features, f"Expected {self.num_features} features. Got: {D}"
        assert train_y.shape[1] == 1, f"train_Y must be a column vector. Got: {train_y.shape}"
        assert 1 <= num_epochs, f"Must take at least 1 gradient step. Got: {num_epochs}"

        nll, grad = self._compute_loss_and_gradient(train_X, train_y)

        # Check gradient using finite difference
        if check_grad:
            original_parameters = np.copy(self.parameters)
            grad_approx = np.zeros(shape=(self.num_classes, self.num_features + 1))
            h = 1e-8

            # Compute finite difference w.r.t. each weight vector component
            for ii in range(self.num_classes):
                for jj in range(self.num_features + 1):
                    self.parameters = np.copy(original_parameters)
                    self.parameters[ii][jj] += h
                    grad_approx[ii][jj] = (self._compute_loss_and_gradient(train_X, train_y)[0] - nll) / h

            # Reset parameters back to original
            self.parameters = np.copy(original_parameters)

            print(f"Negative Log Likelihood: {nll}")
            print(f"Analytic Gradient: {grad.T}")
            print(f"Numerical Gradient: {grad_approx.T}")
            print("The gradients should be nearly identical.")

        # Perform gradient descent
        for epoch_i in range(num_epochs):
            original_parameters = np.copy(self.parameters)
            # Check gradient flow
            if np.linalg.norm(grad) < eps:
                print(f"Gradient is close to 0: {eps}. Terminating gradient descent.")
                break

            # Determine the suitable step size.
            step_size *= 2
            self.parameters = original_parameters - step_size * grad
            E_new, grad_new = self._compute_loss_and_gradient(train_X, train_y)
            assert np.isfinite(E_new), f"Error is NaN/Inf"

            while E_new >= nll and step_size > 0:
                step_size /= 2
                self.parameters = original_parameters - step_size * grad
                E_new, grad_new = self._compute_loss_and_gradient(train_X, train_y)
                assert np.isfinite(E_new), f"Error is NaN/Inf"

            if step_size <= eps:
                print(f"Infinitesimal step: {step_size}. Terminating gradient descent.")
                break

            if verbose:
                print(f"Epoch: {epoch_i}, Step size: {step_size}, Gradient Norm: {np.linalg.norm(grad)}, NLL: {nll}")

            # Update next loss and next gradient
            grad = grad_new
            nll = E_new

    def predict(self, X):
        """ This computes the probability of the K labels given the input X.

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.

        Output:
        - probs (ndarray (shape: (N, K))): A NxK matrix consisting N K-probabilities for each input.
        """
        (N, D) = X.shape
        assert D == self.num_features, f"Expected {self.num_features} features. Got: {D}"

        # Pad 1's for bias term
        X = np.hstack((np.ones(shape=(N, 1), dtype=np.float), X))

        # This receives the probabilities of class 1 given inputs
        probs = softmax(X @ self.parameters.T)
        return probs
