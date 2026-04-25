import numpy as np
from dot_product import dot_product


def norm(A):
    x = 0
    for n in range(A.size):
        x += A[n] * A[n]
    return np.sqrt(x)

# https://en.wikipedia.org/wiki/Cosine_similarity
# cos(θ) = (a · b) / (||a|| * ||b||)
def cosine_similarity(A, B):
    return dot_product(A, B) / (norm(A) * norm(B))

