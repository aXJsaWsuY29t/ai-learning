import numpy as np
from matmul import matmul
from dot_product import dot_product


def test_matmul():
    print("\n=== matmul ===")
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    print(f"A = {A}")
    print(f"B = {B}")

    result = matmul(A, B)
    expected = np.dot(A, B)
    allclose = np.allclose(result, expected)
    print(f"result = {result}")
    print(f"expected = {expected}")
    print(f"allclose = {allclose}")


def test_dot_product():
    print("\n=== dot_product ===")
    A = np.array([1,2,3,4])
    B = np.array([3,5,7,9])
    print(f"A = {A}")
    print(f"B = {B}")

    result = dot_product(A, B)
    expected = np.dot(A, B)

    print(f"result = {result}")
    print(f"expected = {expected}")


test_matmul()
test_dot_product()
