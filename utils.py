"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Exam
B. Chan, D. Fleet
"""

import _pickle as pickle
import numpy as np

def softmax(logits):
    """ This function applies the softmax function to a vector of logits.

    Args:
    - logits (ndarray (shape: (N, K))): A NxK matrix containing N K-dimensional logit vectorss.

    Output:
    - (ndarray (shape: (N, K))): A NxK matrix containing N categorical distributions over K classes.
    """
    e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_logits / np.sum(e_logits, axis=1, keepdims=True)

def load_pickle_dataset(file_path):
    """ This function loads a pickle file given a file path.

    Args:
    - file_path (str): The path of the pickle file

    Output:
    - (dict): A dictionary consisting the dataset content.
    """
    return pickle.load(open(file_path, "rb"))
