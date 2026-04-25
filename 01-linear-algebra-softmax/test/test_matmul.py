import numpy as np
from src.matmul import matmul


def test_matmul_matches_numpy():
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    assert np.allclose(matmul(A, B), np.dot(A, B))


def test_matmul_shape():
    A = np.random.randn(2, 5)
    B = np.random.randn(5, 3)
    assert matmul(A, B).shape == (2, 3)


def test_matmul_incompatible_shapes():
    A = np.random.randn(3, 4)
    B = np.random.randn(3, 2)  # wrong: 4 != 3
    try:
        matmul(A, B)
        assert False, "should have raised"
    except AssertionError:
        pass
