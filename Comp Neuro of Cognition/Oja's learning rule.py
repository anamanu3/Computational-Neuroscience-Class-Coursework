"""
Oja's Rule — working notes

What this file is doing (in normal language):
- We generate correlated data and train a single linear neuron with two rules:
  (1) plain Hebbian learning, and (2) Oja's rule (Hebb + a decay term).
- Hebbian alone makes the weight vector grow without bound. Oja keeps it
  normalized and, in practice, pushes the weights toward the first principal
  component (PC1) of the input covariance (i.e., it does PCA).
- Exercise 1: compare Hebb vs Oja on 2D data (plots show learned directions,
  weight growth, angle to the true PC, etc.).
- Exercise 2: see what happens when we change the learning rate (speed vs
  stability).
- Exercise 3: higher-dimensional data; learn several PCs sequentially using
  Oja + deflation (remove the projection on the found PC and repeat).

How to read the figures quickly:
- “Input Data and Learned Directions”: arrows show the true PC1 and the final
  directions learned by Hebb and Oja.
- “Weight Norm”: if it explodes, Hebb is doing what Hebb does; Oja should stay
  roughly bounded.
- “Angular Error”: angle (in degrees) between the current weight vector and PC1.
  Lower is better; a flat line near zero means we’re aligned with PC1.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("Oja's rule — synaptic plasticity and PCA")
print("="*60)

def simulate_oja_learning(input_data, learning_rate=0.01, n_epochs=1000):
    """Run Oja's rule on input_data and record weight history and norms."""
    n_patterns, n_features = input_data.shape
    w = np.random.randn(n_features) * 0.1  # random small init
    weights_history, weight_norms = [], []

    print(f"training (oja): {n_epochs} epochs")

    for epoch in range(n_epochs):
        for idx in np.random.permutation(n_patterns):
            x = input_data[idx]
            y = np.dot(w, x)
            # Oja: dw = η * y * (x - y*w)
            w += learning_rate * y * (x - y * w)

        weights_history.append(w.copy())
        weight_norms.append(np.linalg.norm(w))
        if (epoch + 1) % 200 == 0:
            print(f"  epoch {epoch + 1:4d} | ‖w‖ = {np.linalg.norm(w):.3f}")

    return np.array(weights_history), np.array(weight_norms)

def simulate_hebbian_learning(input_data, learning_rate=0.01, n_epochs=1000):
    """Run plain Hebbian learning for comparison."""
    n_patterns, n_features = input_data.shape
    w = np.random.randn(n_features) * 0.1
    weights_history, weight_norms = [], []

    print(f"training (hebb): {n_epochs} epochs")

    for epoch in range(n_epochs):
        for idx in np.random.permutation(n_patterns):
            x = input_data[idx]
            y = np.dot(w, x)
            # Hebb: dw = η * y * x
            w += learning_rate * y * x

        weights_history.append(w.copy())
        weight_norms.append(np.linalg.norm(w))
        if (epoch + 1) % 200 == 0:
            print(f"  epoch {epoch + 1:4d} | ‖w‖ = {np.linalg.norm(w):.3f}")

    return np.array(weights_history), np.array(weight_norms)

def exercise_1_oja_vs_hebbian():
    """Compare Oja vs Hebbian on simple 2D data."""
    print("\n" + "-"*50)
    print("exercise 1: oja vs hebbian (2D)")
    print("-"*50)

    # synthetic 2D data with a known covariance
    np.random.seed(42)
    n_samples = 200
    data = np.random.multivariate_normal([0, 0], [[2, 1.5], [1.5, 1]], n_samples)

    # true PCA for reference
    C = np.cov(data.T)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    pc1 = evecs[:, 0]

    print(f"true PC1 ≈ [{pc1[0]:.3f}, {pc1[1]:.3f}] | eigenvalues = [{evals[0]:.3f}, {evals[1]:.3f}]")

    # train both rules
    eta, n_epochs = 0.001, 1000
    w_oja_hist, norm_oja = simulate_oja_learning(data, eta, n_epochs)
    w_hebb_hist, norm_hebb = simulate_hebbian_learning(data, eta, n_epochs)

    # --- plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Oja vs Hebb", fontsize=16, fontweight='bold')

    # 1) data + learned directions
    ax = axes[0, 0]
    ax.scatter(data[:, 0], data[:, 1], s=20, alpha=0.5)
    scale = 3
    ax.arrow(0, 0, *(pc1 * scale), head_width=0.1, head_length=0.1, ec='k', fc='k', lw=2, label='true PC1')
    oja_dir = w_oja_hist[-1] / np.linalg.norm(w_oja_hist[-1]) * scale
    hebb_dir = w_hebb_hist[-1] / np.linalg.norm(w_hebb_hist[-1]) * scale
    ax.arrow(0, 0, *oja_dir, head_width=0.1, head_length=0.1, ec='r', fc='r', lw=2, alpha=0.8, label='oja final')
    ax.arrow(0, 0, *hebb_dir, head_width=0.1, head_length=0.1, ec='g', fc='g', lw=2, alpha=0.8, label='hebb final')
    ax.set_title("data + learned directions")
    ax.axis('equal'); ax.grid(alpha=0.3); ax.legend()

    # 2) weight evolution
    ax = axes[0, 1]
    epochs = np.arange(len(w_oja_hist))
    ax.plot(epochs, w_oja_hist[:, 0], 'r-', lw=2, label='oja w1')
    ax.plot(epochs, w_oja_hist[:, 1], 'r--', lw=2, label='oja w2')
    ax.plot(epochs, w_hebb_hist[:, 0], 'g-', lw=2, label='hebb w1')
    ax.plot(epochs, w_hebb_hist[:, 1], 'g--', lw=2, label='hebb w2')
    ax.axhline(pc1[0], color='k', ls=':', alpha=0.6)
    ax.axhline(pc1[1], color='k', ls=':', alpha=0.6)
    ax.set_title("weights over epochs"); ax.grid(alpha=0.3); ax.legend()

    # 3) weight norms
    ax = axes[0, 2]
    ax.plot(epochs, norm_oja, 'r-', lw=2, label='oja ‖w‖')
    ax.plot(epochs, norm_hebb, 'g-', lw=2, label='hebb ‖w‖')
    ax.axhline(1.0, color='k', ls='--', alpha=0.6)
    ax.set_title("weight norm"); ax.grid(alpha=0.3); ax.legend()
    ax.set_yscale('log')

    # 4) weight-space trajectories
    ax = axes[1, 0]
    ax.plot(w_oja_hist[:, 0], w_oja_hist[:, 1], 'r-', lw=2, alpha=0.8, label='oja')
    ax.plot(w_hebb_hist[:, 0], w_hebb_hist[:, 1], 'g-', lw=2, alpha=0.8, label='hebb')
    ax.plot(w_oja_hist[0, 0], w_oja_hist[0, 1], 'ko', ms=7, label='start')
    ax.plot(w_oja_hist[-1, 0], w_oja_hist[-1, 1], 'r^', ms=8, label='oja end')
    ax.plot(w_hebb_hist[-1, 0], w_hebb_hist[-1, 1], 'g^', ms=8, label='hebb end')
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), 'k--', alpha=0.4)
    ax.arrow(0, 0, *pc1, head_width=0.02, head_length=0.02, ec='k', fc='k', lw=1.5)
    ax.set_title("trajectory in weight space"); ax.axis('equal'); ax.grid(alpha=0.3); ax.legend()

    # helper for angles
    def angle_to_pc(weights, pc):
        ang = []
        for w in weights:
            c = np.dot(w, pc) / (np.linalg.norm(w) * np.linalg.norm(pc))
            c = np.clip(c, -1, 1)
            ang.append(np.degrees(np.arccos(abs(c))))
        return np.array(ang)

    # 5) angular error
    ax = axes[1, 1]
    ang_oja = angle_to_pc(w_oja_hist, pc1)
    ang_hebb = angle_to_pc(w_hebb_hist, pc1)
    ax.plot(epochs, ang_oja, 'r-', lw=2, label='oja')
    ax.plot(epochs, ang_hebb, 'g-', lw=2, label='hebb')
    ax.set_title("angle to PC1 (deg)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_yscale('log')

    # 6) simple summary box
    ax = axes[1, 2]
    txt = (
        f"final angular error (oja):  {ang_oja[-1]:.2f}°\n"
        f"final angular error (hebb): {ang_hebb[-1]:.2f}°\n\n"
        f"final ‖w‖ (oja):  {norm_oja[-1]:.3f}\n"
        f"final ‖w‖ (hebb): {norm_hebb[-1]:.1f}\n\n"
        "notes:\n"
        "- oja aligns with PC1 and keeps ‖w‖ bounded\n"
        "- hebb points toward PC1 but ‖w‖ grows"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
            fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9))
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\nresults: angle(oja→PC1)={ang_oja[-1]:.2f}°, ‖w‖={norm_oja[-1]:.3f} | "
          f"angle(hebb→PC1)={ang_hebb[-1]:.2f}°, ‖w‖={norm_hebb[-1]:.1f}")

def exercise_2_learning_rate_effects():
    """Quick sweep over learning rates for Oja."""
    print("\n" + "-"*50)
    print("exercise 2: learning-rate effects (oja)")
    print("-"*50)

    np.random.seed(123)
    n_samples = 150
    data = np.random.multivariate_normal([0, 0], [[2, 1], [1, 1.5]], n_samples)

    C = np.cov(data.T)
    evals, evecs = np.linalg.eigh(C)
    pc1_true = evecs[:, np.argmax(evals)]

    lrs = [1e-4, 1e-3, 1e-2, 1e-1]
    n_epochs = 600
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Oja: learning-rate sweep", fontsize=16, fontweight='bold')

    for i, lr in enumerate(lrs):
        print(f"  η = {lr}")
        w_hist, norms = simulate_oja_learning(data, lr, n_epochs)
        ep = np.arange(len(w_hist))

        ax = axes[0, i]
        ax.plot(ep, w_hist[:, 0], 'r-', lw=2, label='w1')
        ax.plot(ep, w_hist[:, 1], 'b-', lw=2, label='w2')
        ax.axhline(pc1_true[0], color='r', ls='--', alpha=0.6)
        ax.axhline(pc1_true[1], color='b', ls='--', alpha=0.6)
        ax.set_title(f"η={lr}  (weights)"); ax.grid(alpha=0.3); ax.legend()

        ax = axes[1, i]
        ax.plot(ep, norms, 'k-', lw=2)
        ax.axhline(1.0, color='k', ls='--', alpha=0.6)
        ax.set_title(f"η={lr}  (‖w‖)"); ax.grid(alpha=0.3)

        # quick stability note
        fin = norms[-1]
        fw = w_hist[-1] / np.linalg.norm(w_hist[-1])
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(fw, pc1_true)), 0, 1)))
        print(f"     final angle ≈ {ang:.2f}°, final ‖w‖ ≈ {fin:.3f}")

    plt.tight_layout()
    plt.show()

def exercise_3_multi_dimensional_pca():
    """Sequential PCs in 4D with Oja + deflation."""
    print("\n" + "-"*50)
    print("exercise 3: multi-dimensional PCA (4D)")
    print("-"*50)

    np.random.seed(789)
    n_samples = 250
    base = np.random.randn(n_samples, 4)
    transform = np.array([
        [2.0, 0.5, 0.2, 0.1],
        [0.3, 1.5, 0.4, 0.1],
        [0.1, 0.3, 1.0, 0.2],
        [0.05, 0.1, 0.1, 0.5]
    ])
    data = base @ transform.T

    C = np.cov(data.T)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    print("true eigenvalues:", evals)

    pcs = []
    cur = data.copy()
    for k in range(3):  # first 3 PCs
        print(f"  learning PC{k+1} ...")
        w_hist, _ = simulate_oja_learning(cur, learning_rate=5e-4, n_epochs=800)
        pc = w_hist[-1] / np.linalg.norm(w_hist[-1])
        pcs.append(pc)
        if k < 2:
            proj = cur @ pc
            cur = cur - np.outer(proj, pc)

    # visuals
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Oja + deflation (4D)", fontsize=16, fontweight='bold')

    # eigenvalues vs learned (variance along learned PCs)
    ax = axes[0, 0]
    learned_vals = []
    for pc in pcs:
        learned_vals.append(np.var(data @ pc))
    x = np.arange(len(evals))
    w = 0.35
    ax.bar(x - w/2, evals, w, label='true', alpha=0.7)
    ax.bar(x[:len(learned_vals)] + w/2, learned_vals, w, label='learned', alpha=0.7)
    ax.set_title("eigenvalue recovery"); ax.set_xticks(x); ax.set_xticklabels([f"PC{i+1}" for i in x])
    ax.grid(alpha=0.3, axis='y'); ax.legend()

    # correlations PC1..PC3
    for i in range(3):
        ax = axes[0, i+1] if i < 2 else axes[1, 0]
        tpc, lpc = evecs[:, i], pcs[i]
        corr = max(np.corrcoef(tpc, lpc)[0, 1], np.corrcoef(tpc, -lpc)[0, 1])
        ax.scatter(tpc, lpc, alpha=0.7, s=50)
        mn, mx = min(tpc.min(), lpc.min()), max(tpc.max(), lpc.max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
        ax.set_title(f"PC{i+1} corr  r={corr:.3f}")
        ax.grid(alpha=0.3)

    # cumulative variance explained
    ax = axes[1, 1]
    total = np.sum(evals)
    cum_true = np.cumsum(evals) / total * 100
    cum_learn = np.cumsum(learned_vals + [0]) / total * 100
    ax.plot(range(1, len(evals)+1), cum_true, 'bo-', lw=3, ms=7, label='true')
    ax.plot(range(1, len(learned_vals)+1), cum_learn[:len(learned_vals)], 'ro-', lw=3, ms=7, label='oja')
    ax.set_title("variance explained"); ax.grid(alpha=0.3); ax.legend()

    # small summary
    ax = axes[1, 2]
    ang_err = []
    for i in range(3):
        c = abs(np.dot(pcs[i], evecs[:, i]))
        ang_err.append(np.degrees(np.arccos(np.clip(c, 0, 1))))
    txt = (
        f"PC angle errors (deg): {', '.join(f'{a:.2f}' for a in ang_err)}\n"
        f"learned eigenvalue ratios:\n"
        f"  PC1 {learned_vals[0]/evals[0]:.3f}, PC2 {learned_vals[1]/evals[1]:.3f}, PC3 {learned_vals[2]/evals[2]:.3f}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
            fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9))
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"angles to true PCs (deg): {', '.join(f'{a:.2f}' for a in ang_err)}")

if __name__ == "__main__":
    print("starting oja experiments...")
    exercise_1_oja_vs_hebbian()
    exercise_2_learning_rate_effects()
    exercise_3_multi_dimensional_pca()
    print("done.")