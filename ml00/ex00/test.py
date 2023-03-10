from matrix import Matrix
from matrix import Vector
import sys

if __name__ == "__main__":
    v0 = Matrix([[1,2,3], [1, 2, 3], 3])
    print((v0 * v1).__repr__())

    print((v0 - v1).__repr__())
    print(v0.T().__repr__())
    shmat1 = (3, 2)
    print("Matrix from shape : ", shmat1)
    mat1 = Matrix(shmat1)
    print(mat1.__str__())
    print(mat1.__repr__())

    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1)
    print("shape :", m1.shape)
    print("transpose :")
    print(m1.T())
    print("shape :", m1.T().shape)

    print("on the other way")

    m2 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m2)
    print("shape :", m2.shape)
    print("transpose :")
    print(m2.T())
    print("shape :", m2.T().shape)

    m3 = Matrix([[1., 2., 3.], [1., 2., 3.]])
    print("add")
    print(m2 + m3)
    print("radd")
    print(m2.__radd__(m3))
    print("sub")
    print(m2 - m3)
    print("rsub")
    print(m2.__rsub__(m3))
    print("truediv by 2")
    print(m2 / 2)
    print("mult by 2")
    print(m2 * 2)
    mm1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
    mm2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

    mt1= Matrix([[1.0, 2.0]])
    mt2 = Matrix([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    print("mm1 * mm2")
    print(mm1 * mm2)
    print("mt1 * mt2")
    print(mt1 * mt2)
    print("rmult")
    print(mt2.__rmult__(mt1))

    print(Vector((8,1)))
    v1 = Vector([[0.0], [1.0], [2.0]])
    v2 = Vector([[3.0], [8.0], [6.0]])
    print(v1.dot(v2))
    rse = v1 + v2
    print(rse)
    print(rse.__repr__())
    print("matrix by vector multiplication : ")

    mv1 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    mv2 = Vector([[1], [2], [3]])

    mv1v2 = mv1 * mv2
    print(mv1v2)