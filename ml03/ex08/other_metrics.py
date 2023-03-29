import numpy as np
import sys

def check_matix(mat):
    if not (isinstance(mat, np.ndarray)):
        return False
    if mat.dtype != "int64" and mat.dtype != "float64" and not (mat.dtype == 'U8'):
        return False
    if len(mat.shape) == 1:
        mat = np.atleast_2d(mat).T
    if len(mat.shape) != 2:
        return False
    if (mat.size == 0):
        return False
    return True

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a fl:oat.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """

    if not check_matix(y) or not check_matix(y_hat):
        return None
    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
    if len(y_hat.shape) == 1:
        y_hat = np.atleast_2d(y_hat).T
    if y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None

    correct = 0

    for idx in range(y.shape[0]):
        if (y[idx] == y_hat[idx]):
            correct = correct + 1
        


    return (correct / y.shape[0])

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matix(y) or not check_matix(y_hat):
        return None
    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
    if len(y_hat.shape) == 1:
        y_hat = np.atleast_2d(y_hat).T
    if y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    if not type(pos_label) is str and not type(pos_label) is int:
        return None 

    st = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' : 0}

    for idx in range(y.shape[0]):
        if (y_hat[idx] == y[idx]):
            if y[idx] == pos_label:
                st['tp'] = st['tp'] + 1
            else:
                st['tn'] = st['tn'] + 1
        else:
            if y[idx] == pos_label:
                st['fn'] = st['fn'] + 1
            else:
                st['fp'] = st['fp'] + 1

    if st['tp'] == 0:
        return float(0)

    if (st['tp'] + st['fp'] == 0):
        return float('inf')

    return(st['tp'] / (st['tp'] + st['fp']))


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matix(y) or not check_matix(y_hat):
        return None
    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
    if len(y_hat.shape) == 1:
        y_hat = np.atleast_2d(y_hat).T
    if y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    if not type(pos_label) is str and not type(pos_label) is int:
        return None 

    st = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' : 0}

    for idx in range(y.shape[0]):
        if (y_hat[idx] == y[idx]):
            if y[idx] == pos_label:
                st['tp'] = st['tp'] + 1
            else:
                st['tn'] = st['tn'] + 1
        else:
            if y[idx] == pos_label:
                st['fn'] = st['fn'] + 1
            else:
                st['fp'] = st['fp'] + 1

    if st['tp'] == 0:
        return float(0)

    if (st['tp'] + st['fn'] == 0):
        return float('inf')
    return (st['tp'] / (st['tp'] + st['fn']))


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matix(y) or not check_matix(y_hat):
        return None
    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
    if len(y_hat.shape) == 1:
        y_hat = np.atleast_2d(y_hat).T
    if y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    if not type(pos_label) is str and not type(pos_label) is int:
        return None 

    st = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' : 0}

    for idx in range(y.shape[0]):
        if (y_hat[idx] == y[idx]):
            if y[idx] == pos_label:
                st['tp'] = st['tp'] + 1
            else:
                st['tn'] = st['tn'] + 1
        else:
            if y[idx] == pos_label:
                st['fn'] = st['fn'] + 1
            else:
                st['fp'] = st['fp'] + 1


    prec = st['tp'] / (st['tp'] + st['fp'])
    reca = st['tp'] / (st['tp'] + st['fn'])

    if (prec == 0 or reca == 0):
        return float(0)
    if (prec + reca == 0):
        return float('inf')

    return ((2 * prec * reca) / (prec + reca))


if __name__ == "__main__":

    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

    print("Exemple 1\n")

    print("Accurency")
    print(accuracy_score_(y, y_hat))
    print("Precision")
    print(precision_score_(y, y_hat))
    print("Recall")
    print(recall_score_(y, y_hat))
    print("F1 score")
    print(f1_score_(y, y_hat))

    print("Exemple 2\n")

    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])


    print("Accurency")
    print(accuracy_score_(y, y_hat))
    print("Precision")
    print(precision_score_(y, y_hat, pos_label='dog'))
    print("Recall")
    print(recall_score_(y, y_hat, pos_label='dog'))
    print("F1 score")
    print(f1_score_(y, y_hat, pos_label='dog'))

    print("Exemple 3\n")

    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    print("Precision")
    print(precision_score_(y, y_hat, pos_label='norminet'))
    print("Recall")
    print(recall_score_(y, y_hat, pos_label='norminet'))
    print("F1 score")
    print(f1_score_(y, y_hat, pos_label='norminet'))