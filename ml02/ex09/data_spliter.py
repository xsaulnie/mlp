import numpy as np

def check_matix(mat):
    if not (isinstance(mat, np.ndarray)):
        return False
    if mat.dtype != "int64" and mat.dtype != "float64":
        return False
    if len(mat.shape) != 2:
        return False
    if (mat.size == 0):
        return False
    return True

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if not (check_matix(x) or not check_matix(y)):
        return None
    if not(type(proportion) is float):
        return None
    if (proportion > 1 or proportion < 0):
        return None
    if x.shape[0] != y.shape[0]:
        return None

    data = np.concatenate((x, y), axis = 1)
    np.random.shuffle(data)

    nb = int(proportion * data.shape[0])

    return((data[0:nb,:-1], data[nb:,:-1], (data[0:nb,-1:]).T[0], (data[nb:,-1:]).T[0]))

if __name__ == "__main__":

    print("Exemple 1")
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    print("x1")
    print(x1)
    print("y")
    print(y)
    print("data_spliter(x1, y, 0.8)")
    print(data_spliter(x1, y, 0.8))

    print("Exemple 2")
    print("data_spliter(x1, y, 0.5)")
    print(data_spliter(x1, y, 0.5))

    x2 = np.array([[ 1, 42], [300, 10], [ 59, 1], [300, 59], [ 10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

    print("Exemple 3")
    print("x2")
    print(x2)
    print("y")
    print(y)
    print("data_spliter(x2, y, 0.8)")
    print(data_spliter(x2, y, 0.8))

    print("Exemple 4")
    print("data_spliter(x2, y, 0.8)")
    print(data_spliter(x2, y, 0.5))




