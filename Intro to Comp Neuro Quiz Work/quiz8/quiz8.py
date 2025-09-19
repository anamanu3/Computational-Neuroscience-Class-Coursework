"""
Oja’s rule demo (quiz7)

- Load 2-D data from the pickle file and center it (zero mean).
- Initialize a random weight vector w and update it with Oja’s
  learning rule over many iterations.
- At convergence, w should point in the direction of the first
  principal component (PCA1) and have a stable norm ~ 1/√α.
- Finally, compute PCA directly via the covariance matrix and
  check how close w is by taking the cosine of the angle.
"""



import pickle
import numpy as np
from pathlib import Path

data_path = Path("/Users/anamariamanu/Intro to Comp Neuro/quiz7/_7bfd5defa66c4d019fdb4bd6af2a62b5_c10p1.pickle")

with open(data_path, "rb") as f:
    raw = pickle.load(f)

X = np.asarray(raw['c10p1'], dtype=float)  

X = X - X.mean(axis=0, keepdims=True)
N = X.shape[0]

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

#eigenvector
C = (X.T @ X) / N
eigvals, eigvecs = np.linalg.eig(C)
pca1 = eigvecs[:, np.argmax(eigvals.real)].real
pca1 /= np.linalg.norm(pca1) + 1e-12

cosang = abs(np.dot(w/np.linalg.norm(w), pca1))
print("cos(angle(w, PCA1)) =", cosang)