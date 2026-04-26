import numpy as np
from matmul import matmul
from dot_product import dot_product
from cosine_similarity import cosine_similarity
from softmax import softmax


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


def test_cosine_similarity():
    print("\n=== cosine_similarity ===")
    A = np.array([2, 3, 4, 5])
    B = np.array([3, 4, 5, 6])
    print(f"A = {A}")
    print(f"B = {B}")

    result = cosine_similarity(A, B)
    expected = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    print(f"result = {result}")
    print(f"expected = {expected}")


def test_softmax():
    print("\n=== softmax ===")
    A = np.array([2.0, 1.0, 0.5])
    result = softmax(A)
    expected = np.exp(A) / np.sum(np.exp(A))
    allclose = np.allclose(result, expected)

    print(f"sum = {sum(result)}, must be 1.0")
    print(f"result = {result}")
    print(f"expected = {expected}")
    print(f"allclose = {allclose}")

    index_max_a = np.argmax(A)
    index_max_result = result.index(max(result))
    print(f"max() index in A = {index_max_a}, in result = {index_max_result}")

test_matmul()
test_dot_product()
test_cosine_similarity()
test_softmax()

