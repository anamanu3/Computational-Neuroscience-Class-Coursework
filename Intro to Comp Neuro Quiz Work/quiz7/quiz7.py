"""
Q7 — Oja’s learning rule on real data

- Load a dataset (matrix X) from a pickle file.
- Center the data (subtract the mean of each column).
- Run Oja’s learning rule to update a 2-D weight vector w:
    w ← w + η·dt·(v·u − α·v²·w),  where v = w·u.
- After many steps, w should stabilize with norm ≈ 1/√α.
- Compare the learned direction to the first principal component
  of X (via eigen-decomposition of the covariance matrix).
- Print the cosine of the angle between w and PC1 as a measure
  of how well Oja’s rule recovered the leading eigenvector.
"""


#Q7:
import pickle
import numpy as np
from pathlib import Path

data_path = Path("/Users/anamariamanu/Intro to Comp Neuro/quiz7/_7bfd5defa66c4d019fdb4bd6af2a62b5_c10p1.pickle")

#dict
with open(data_path, "rb") as f:
    raw = pickle.load(f)

X = np.asarray(raw['c10p1'], dtype=float) 

X = X - X.mean(axis=0, keepdims=True)
N = X.shape[0]

#para:
eta = 1.0
alpha = 1.0
dt = 0.01
steps = 100_000

rng = np.random.default_rng(0)
w = rng.normal(size=2)
w /= np.linalg.norm(w) + 1e-12

for t in range(steps):
    u = X[t % N]       
    v = float(np.dot(u, w))   
    w += dt * eta * (v * u - alpha * (v ** 2) * w)

print("Learned weight vector w:", w)
print("||w||:", np.linalg.norm(w))
print("Expected stable norm ~", 1/np.sqrt(alpha))

#eigenfactor
C = (X.T @ X) / N
eigvals, eigvecs = np.linalg.eig(C)
pca1 = eigvecs[:, np.argmax(eigvals.real)].real
pca1 /= np.linalg.norm(pca1) + 1e-12

cosang = abs(np.dot(w/np.linalg.norm(w), pca1))
print("cos(angle(w, PCA1)) =", cosang)