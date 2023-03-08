class Matrix:
    def __init__(self, arg):
        if (type(arg) is tuple):
            if (len(arg) != 2):
                raise ValueError("Matrix : shape constructor wrong dimension")
                return
            self.shape = arg
            self.data = []

            for x in range(self.shape[0]):
                lin = []
                for y in range(self.shape[1]):
                    lin.append(0)
                self.data.append(lin)
            return

        elif (type(arg) is list):
            for x in arg:
                if not type(arg) is list:
                    raise TypeError("Matrix: list constructor must be list of list")
                    return
            dim = len(arg[0])
            for x in arg:
                if dim != len(x):
                    raise ValueError("Matrix: wrong dimensions on list constructor")
                    return
                for elem in x:
                    if (not (type(elem) is int or type(elem) is float)):
                        raise TypeError("Matrix: Elements must be float or int")
                        return

            self.shape = (len(arg), len(arg[0]))
            self.data = []

            for x in range(self.shape[0]):
                lin = []
                for y in range(self.shape[1]):
                    lin.append(arg[x][y])
                self.data.append(lin)
            return
        else:
            raise TypeError("Matrix: wrong type on constructor")
            return

    def T(self):
        ret = Matrix((self.shape[1], self.shape[0]))

        for x in range(self.shape[1]):
            for y in range(self.shape[0]):
                ret.data[x][y] = self.data[y][x]
        return ret
    
    def __add__(self, mat):
        if (not isinstance(mat, Matrix)):
            raise NotImplementedError("Matrix : __add__ operation not define on this type")
            return
        if mat.shape != self.shape:
            raise ValueError("Matrix : __add__ wrong dimension")
        ret = []
        for x1, x2 in zip(self.data, mat.data):
            lin = []
            for y1, y2 in zip(x1, x2):
                lin.append(y1 + y2)
            ret.append(lin)
        return Matrix(ret)

    def __radd__(self, mat):
        if (not isinstance(mat, Matrix)):
            raise NotImplementedError("Matrix : __radd__ operation not define on this type")
            return
        if mat.shape != self.shape:
            raise ValueError("Matrix : __radd__ wrong dimension")
        return(self.__add__(mat))

    def __sub__(self, mat):
        if (not isinstance(mat, Matrix)):
            raise NotImplementedError("Matrix : __sub__ operation not define on this type")
            return
        if mat.shape != self.shape:
            raise ValueError("Matrix : __sub__ wrong dimension")
            return
        ret = []
        for x1, x2 in zip(self.data, mat.data):
            lin = []
            for y1, y2 in zip(x1, x2):
                lin.append(y1 - y2)
            ret.append(lin)
        return Matrix(ret)

    def __rsub__(self, mat):
        if (not isinstance(mat, Matrix)):
            raise NotImplementedError("Matrix : __rsub__ operation not define on this type")
            return
        if mat.shape != self.shape:
            raise ValueError("Matrix : __rsub__ wrong dimension")
        return(mat.__sub__(self))

    def __truediv__(self, scalar):
        if (type(scalar) is not int and type(scalar) is not float):
            raise NotImplementedError("Matrix: __truediv__ operation not define on this type")
            return
        if scalar == 0 or scalar == 0.0:
            raise ZeroDivisionError("Matrix : __truediv__ division by 0")
            return
        ret = []
        for x in self.data:
            lin = []
            for y in x:
                lin.append(y / scalar)
            ret.append(lin)
        return ret
    def __rtruediv__(self, arg):
        raise NotImplementedError("Matrix: __rtruediv__ division by a Matrix not defined")

    def __mul__(self, arg):
        if (type(arg) is int or type(arg) is float):
            ret = []
            for x  in self.data:
                lin = []
                for y in x:
                    lin.append(y * arg)
                ret.append(lin)
            return (ret)
        elif (isinstance(arg, Vector)):
            #print(arg.shape, self.shape)
            if not (arg.shape[0] == self.shape[1]):
                raise ValueError("Matrix: __mult__ on vector, wrong dimension")
                return
            if (type(self) is Vector):
                raise NotImplementedError("Vector __mult__ beetwin vectors")
                return
            ret = Vector((self.shape[0], 1))
            for idx, lin in enumerate(self.data):
                sumdot = 0
                for x, y in zip(lin, arg.data):
                    #print(x, y)
                    sumdot = sumdot + x * y[0]
                ret.data[idx] = [sumdot]
            return ret
        elif (isinstance(arg, Matrix)):
            if not (arg.shape[0] == self.shape[1]):
                raise ValueError("Matrix: __mult__ multiplication on wrong dimensions")
                return
            ret = Matrix((self.shape[0], arg.shape[1]))

            for cx in range(self.shape[0]):
                for cy in range(arg.shape[1]):
                    tot = 0
                    for k in range(self.shape[1]):
                        tot = tot + self.data[cx][k] * arg.data[k][cy]
                    ret.data[cx][cy] = tot
            return (ret)
        else:
            raise NotImplementedError("Matrix: __mult__ no multiplication implemented on this type")
            return
    def __rmult__(self, arg):
        if (type(arg) is int or type(arg) is float):
            ret = []
            for x in self.data:
                lin = []
                for y in x:
                    lin.append(y * arg)
                ret.append(lin)
            return (ret)
        elif isinstance(arg, Matrix):
            if not (self.shape[0] == arg.shape[1]):
                raise ValueError("Matrix: __rmult__ multiplication on wrong dimensions")
                return
            ret = Matrix((arg.shape[0], self.shape[1]))
            for cx in range(arg.shape[0]):
                for cy in range(self.shape[1]):
                    tot = 0
                    for k in range(arg.shape[1]):
                        tot = tot + arg.data[cx][k] * self.data[k][cy]
                    ret.data[cx][cy] = tot
            return (ret)
        else:
            raise NotImplementedError("Matrix: __rmult__ no multiplication implemented on this type")

    def __str__(self):
        res = ("\n ".join('{},'.format(elem) for elem in self.data))
        return "[" + res[:-1] + "]"

    def __repr__(self):
        res = ("".join('{}, '.format(elem) for elem in self.data))
        return "Matrix([" +  res[:-2] + "])"

class Vector(Matrix):
    def __init__(self, arg):
        if (type(arg) is tuple):
            if (len(arg) != 2):
                raise ValueError("Vector : shape constructor wrong dimension")
                return
            if not arg[0] == 1 and not arg[1] == 1:
                raise ValueError("Vector : constructor shape does not correspond to a vector")
            self.shape = arg
            self.data = []

            for x in range(self.shape[0]):
                lin = []
                for y in range(self.shape[1]):
                    lin.append(0)
                self.data.append(lin)
            return

        elif (type(arg) is list):
            for x in arg:
                if not type(arg) is list:
                    raise TypeError("Vector: list constructor must be list of list")
                    return
            dim = len(arg[0])
            for x in arg:

                if dim != len(x):
                    raise ValueError("Vector: wrong dimensions on list constructor")
                    return
                for elem in x:
                    if (not type(elem) is int and not type(elem) is float):
                        raise TypeError("Vector: Elements must be float or int")
            if not len(arg) == 1 and not len(arg[0]) == 1:
                raise ValueError("Vector: constructor list does not correspond to a vector")
            self.shape = (len(arg), len(arg[0]))
            self.data = []

            for x in range(self.shape[0]):
                lin = []
                for y in range(self.shape[1]):
                    lin.append(arg[x][y])
                self.data.append(lin)
            return
        else:
            raise TypeError("Vector: wrong type on constructor")
            return
        
    def __repr__(self):
        res = ("".join('{}, '.format(elem) for elem in self.data))
        return "Vector([" +  res[:-2] + "])"

    def dot(self, vec):
        if(not isinstance(vec, Vector)):
            raise TypeError("Vector: dot product beetween vectors")
            return
        if (self.shape != vec.shape):
            raise ValueError("Vector: dot product dimensions differ")
            return
        ret = 0
        if (self.shape[0] == 1):
            for x, y in zip(self.data[0], vec.data[0]):
                ret = ret + x * y
            return ret
        for x, y in zip(self.data, vec.data):
            ret = ret + x[0] * y[0]
        return ret

    def __add__(self, vec):
        return Vector((super().__add__(vec)).data)

    