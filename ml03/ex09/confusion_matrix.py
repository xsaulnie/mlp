import numpy as np
import pandas as pd

def check_matrix(mat):
    if not isinstance(mat, np.ndarray):
        return False
    if mat.dtype != "U8":
        return False
    if len(mat.shape) != 2:
        return False
    if (mat.size == 0):
        return False
    if (mat.shape[1] != 1):
        return False
    return True

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
        Compute confusion matrix to evaluate the accuhracy of a classification.
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
    if not check_matrix(y_true) or not check_matrix(y_hat):
        return None
    if not labels is None and not type(labels) is list:
        return None
    if not labels is None:
        for elem in labels:
            if not type(elem) is str:
                return None
    if not df_option is True and not df_option is False:
        return None

    listlabels = []
    for lin in range(y_true.shape[0]):
        if not y_true[lin][0] in listlabels:
            listlabels.append(y_true[lin][0])
    for lin in range(y_hat.shape[0]):
        if not y_hat[lin][0] in listlabels:
            listlabels.append(y_hat[lin][0])
    
    listlabel = []
    if not labels is None:
        for elem in listlabels:
            if elem in labels:
                listlabel.append(elem)
    else:
        listlabel = listlabels

    listlabel.sort()

    ret = []
    for label_true in listlabel:
        row = []
        for label_pred in listlabel:
            num = 0
            for idx in range(y_true.shape[0]):
                if (y_hat[idx] == label_pred and y_true[idx] == label_true):
                    num = num + 1
            row.append(num)
        ret.append(row)
    #ret = np.array(ret)
    if df_option is True:
        return pd.DataFrame(data=ret, index=listlabel, columns=listlabel)
    return(np.array(ret))
    

if __name__ == "__main__":
    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
    print("Exemple from subject\n\n")
    print("Exemple 1")
    print(confusion_matrix_(y, y_hat), end='\n\n')
    print("Exemple 2")
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']), end='\n\n')

    print("Exemple 3")
    print(confusion_matrix_(y, y_hat, df_option=True), end='\n\n')
    print("Exemple 4")
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet'], df_option=True))
