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

from logistic_regression import LogisticRegression
from utils import load_pickle_dataset

def train(train_X,
          train_y,
          val_X=None,
          val_y=None,
          factor=1,
          bias=0,
          num_epochs=1000,
          step_size=1e-3,
          check_grad=False,
          verbose=False):
    """ This function trains a logistic regression model on the given training  data.

    Args:
    - train_X (ndarray (shape: (N, D))): A NxD matrix containing N D-dimensional training inputs.
    - train_y (ndarray (shape: (N, 1))): A N-column vector containing N scalar training outputs (labels).
    - val_X (ndarray (shape: (M, D))): A NxD matrix containing M D-dimensional validation inputs.
    - val_y (ndarray (shape: (M, 1))): A N-column vector containing M scalar validation outputs (labels).

    Initialization Args:
    - factor (float): A constant factor to scale the initial weights.
    - bias (float): The bias value

    Learning Args:
    - num_epochs (int): Number of gradient descent steps
                        NOTE: 1 <= num_epochs
    - step_size (float): Gradient descent step size
    - check_grad (bool): Whether or not to check gradient using finite difference.
    - verbose (bool): Whether or not to print gradient information for every step.
    """
    train_accuracy = 0
    # ====================================================
    # TODO: Implement your solution within the box
    # Step 1: Initialize model and initialize weights

    # Step 2: Train the model

    # Step 3: Evaluate training performance

    # ====================================================
    train_preds = np.argmax(train_probs, axis=1)
    train_accuracy = 100 * np.mean(train_preds == train_y.flatten())
    print("Training Accuracy: {}%".format(train_accuracy))

    if val_X is not None and val_y is not None:
        validation_accuracy = 0
        # ====================================================
        # TODO: Implement your solution within the box
        # Evaluate validation performance

        # ====================================================
        val_preds = np.argmax(val_probs, axis=1)
        validation_accuracy = 100 * np.mean(val_preds == val_y.flatten())
        print("Validation Accuracy: {}%".format(validation_accuracy))


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    # Support orchid, generic_1, generic_2
    dataset = "orchid"

    assert dataset in ("orchid", "generic_1", "generic_2"), f"Invalid dataset: {dataset}"

    dataset_path = f"./datasets/{dataset}.pkl"
    data = load_pickle_dataset(dataset_path)

    train_X = data['train_X']
    train_y = data['train_y']
    val_X = val_y = None
    if 'val_X' in data and 'val_y' in data:
        val_X = data['val_X']
        val_y = data['val_y']

    # Hyperparameters
    # NOTE: This is definitely not the best way to pass all your hyperparameters.
    #       We can usually use a configuration file to specify these.
    factor = 1
    bias = 0
    num_epochs = 1000
    step_size = 1e-3
    check_grad = False
    verbose = False

    train(train_X=train_X,
          train_y=train_y,
          val_X=val_X,
          val_y=val_y,
          factor=factor,
          bias=bias,
          num_epochs=num_epochs,
          step_size=step_size,
          check_grad=check_grad,
          verbose=verbose)
