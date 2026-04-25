import numpy as np

# https://en.wikipedia.org/wiki/Matrix_multiplication
def matmul(A, B):
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    assert a_cols == b_rows, "incompatible shapes"

    C = np.zeros((a_rows, b_cols))

    for i in range(a_rows):
        for j in range(b_cols):

            for k in range(a_cols):
                C[i, j] += A[i, k] * B[k, j]

    return C

# test
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)

result = matmul(A, B)
expected = np.dot(A, B)
allclose = np.allclose(result, expected)

print(f"result = {result}")
print(f"expected = {expected}")
print(f"allclose = {allclose}")