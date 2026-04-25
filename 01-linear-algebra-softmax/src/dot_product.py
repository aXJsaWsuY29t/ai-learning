import numpy as np

# https://en.wikipedia.org/wiki/Dot_product
def dot_product(A, B):
    assert A.size == B.size, "incompatible vectors"

    res = 0
    for x in range(A.size):
        res += A[x] * B[x]

    return res
