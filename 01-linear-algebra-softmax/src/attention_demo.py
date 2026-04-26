import numpy as np
from dot_product import dot_product
from cosine_similarity import cosine_similarity
from softmax import softmax

np.random.seed(42)
vectors = [np.random.randn(4) for _ in range(4)]
print(f"\nVectors:")
for i, row in enumerate(vectors):
    print(f"  vector {i}: {[round(w, 3) for w in row]}")

d_k = len(vectors[0])
vectors_dp = [[dot_product(vectors[j], vectors[i]) / np.sqrt(d_k) for i in range(len(vectors))] for j in range(len(vectors))]
print(f"\nDot products:")
for i, row in enumerate(vectors_dp):
    print(f"  vector {i}: {[round(w, 3) for w in row]}")

vectors_softmax = [softmax(vectors_dp[i]) for i in range(len(vectors_dp))]
print(f"\nAttention weights (softmax):")
for i, row in enumerate(vectors_softmax):
    print(f"  vector {i}: {[round(w, 3) for w in row]}")

output = []
for i in range(len(vectors)):
    weighted = np.zeros(len(vectors[0]))
    for j in range(len(vectors)):
        weighted += vectors_softmax[i][j] * vectors[j]
    output.append(weighted)

print(f"\nOutput vectors (after attention):")
for i, v in enumerate(output):
    print(f"  output {i}: {np.round(v, 3)}")
