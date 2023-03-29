import numpy as np
import pandas as pd

def confusion_matrix_(y_true, y_hat, labels=None):
    """
        Compute confusion matrix to evaluate the accuracy of a classification.
        Args:
            y:a numpy.array for the correct labels
            y_hat:a numpy.array for the predicted labels
            labels: optional, a list of labels to index the matrix.
            This may be used to reorder or select a subset of labels. (default=None)
            df_option: optional, if set to True the function will return a pandas DataFrame
            instead of a numpy array. (default=False)
        Return:
            The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
            None if any error.
        Raises:
            This function should not raise any Exception.
    """
    print("confuse")

if __name__ == "__main__":
    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])

    print(np.concatenate((y, y_hat), axis=1))