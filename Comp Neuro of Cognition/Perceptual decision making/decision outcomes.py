"""
Exercise 5 – Network Attractors and Bistability (Wong–Wang model)

What I’m doing here:
- Try out lots of starting conditions and see where the system ends up (phase space).
- Show bistability: same stimulus can lead to different outcomes depending on start.
- Add noise and see how much it messes with the result.
- Slowly ramp coherence up and down to check for hysteresis (path dependence).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("="*60)
print("EXERCISE 5: NETWORK ATTRACTORS AND BISTABILITY")
print("="*60)

# small epsilon to avoid divisions/ranges hitting 0 exactly
_EPS = 1e-9
_rng = np.random.default_rng(7)  # reproducible randomness

def _finite(a):
    """Return array with non-finite entries dropped (for plotting)."""
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(a)
    return a[mask], mask

def _clip01(a):
    """Keep state variables in [0, 1] so they don't wander off."""
    return np.clip(a, 0.0, 1.0)

def _safe_plot(ax, x, y, **kw):
    """Plot only finite points and pin axes to sane ranges."""
    x_f, m1 = _finite(x)
    y_f = np.asarray(y, float)[m1]
    y_f, m2 = _finite(y_f)
    x_f = x_f[m2]
    if x_f.size and y_f.size:
        ax.plot(x_f, y_f, **kw)
    # fixed, finite window (these states are naturally in 0..~0.8)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3)

class WongWangNetwork:
    def __init__(self):
        # Parameters chosen to give attractor dynamics
        self.w_plus = 1.8    # self-excitation
        self.w_minus = 1.2   # cross-inhibition
        self.w_I = 1.0       # global inhibition
        self.tau_s = 60.0    # time constant
        self.I0 = 0.3        # background input
        self.JA_ext = 0.005  # external coupling
        self.mu0 = 30.0      # base input rate
        self.gamma = 0.641   # gain

        print("Wong–Wang network set up (for attractor demos)")

def wong_wang_dynamics(state, t, coherence, net, noise=0.0):
    """Single step of Wong–Wang dynamics (kept simple)."""
    s1, s2, sI = state
    s1, s2, sI = _clip01(s1), _clip01(s2), _clip01(sI)

    # external drive depends on coherence
    c = float(coherence) / 100.0
    nu1 = net.mu0 * (1 + c)
    nu2 = net.mu0 * (1 - c)

    I_ext1 = net.JA_ext * nu1
    I_ext2 = net.JA_ext * nu2

    # recurrent + inhibitory input
    I1 = net.I0 + I_ext1 + net.w_plus*s1 - net.w_minus*s2 - net.w_I*sI
    I2 = net.I0 + I_ext2 + net.w_plus*s2 - net.w_minus*s1 - net.w_I*sI
    I_I = net.I0 + net.gamma*(s1+s2)

    # simple rectified rates
    r1 = max(0.0, I1*100.0)
    r2 = max(0.0, I2*100.0)
    rI = max(0.0, I_I*50.0)

    # update rules
    ds1 = (-s1 + net.gamma*(r1/100.0)*(1.0 - s1)) / (net.tau_s + _EPS)
    ds2 = (-s2 + net.gamma*(r2/100.0)*(1.0 - s2)) / (net.tau_s + _EPS)
    dsI = (-sI + rI/50.0) / (net.tau_s + _EPS)

    # add optional noise (small, zero-mean)
    if noise > 0:
        ds1 += float(noise) * _rng.normal()
        ds2 += float(noise) * _rng.normal()
        dsI += 0.5 * float(noise) * _rng.normal()

    return [ds1, ds2, dsI]

def explore_phase_space():
    """Run a bunch of ICs at different coherences and plot where they go."""
    net = WongWangNetwork()
    coherences = [0, 16, 32]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, coh in enumerate(coherences):
        ax = axes[idx]
        print(f"\nCoherence {coh}%:")

        for _ in range(25):
            # random initial state
            ic = [_rng.uniform(0.05, 0.7), _rng.uniform(0.05, 0.7), 0.1]
            t = np.linspace(0, 1500, 300)

            try:
                sol = odeint(wong_wang_dynamics, ic, t, args=(coh, net, 0.02), mxstep=5000)
                s1, s2, _ = sol.T
            except Exception as e:
                # if the solver stumbles, skip this trajectory
                print("  (skipped a bad trajectory:", e, ")")
                continue

            # outcome color
            diff = (s1[-1] - s2[-1]) if np.isfinite(s1[-1]) and np.isfinite(s2[-1]) else 0.0
            if diff > 0.1:
                color = 'red'
            elif diff < -0.1:
                color = 'blue'
            else:
                color = 'gray'

            _safe_plot(ax, s1, s2, color=color, alpha=0.7, lw=1.5)

        ax.set_title(f'Coherence {coh}%')
        ax.set_xlabel('s1')
        ax.set_ylabel('s2')

    plt.tight_layout()
    plt.show()

def demonstrate_bistability():
    """Pick a few ICs at 0% coherence and see who wins."""
    net = WongWangNetwork()
    coh = 0

    ICs = [
        [0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [0.1, 0.3, 0.1],
        [0.4, 0.2, 0.1], [0.2, 0.4, 0.1], [0.3, 0.3, 0.1]
    ]
    labels = ['low sym', 'bias→1', 'bias→2', 'strong→1', 'strong→2', 'high sym']

    fig, ax = plt.subplots(figsize=(7, 7))
    for ic, label in zip(ICs, labels):
        t = np.linspace(0, 2000, 500)
        try:
            sol = odeint(wong_wang_dynamics, ic, t, args=(coh, net, 0.03), mxstep=5000)
            s1, s2, _ = sol.T
        except Exception as e:
            print("  (skipped)", label, ":", e)
            continue

        _safe_plot(ax, s1, s2, alpha=0.85, lw=2, label=label)

    ax.set_title("Bistability (0% coherence)")
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.legend()
    plt.tight_layout()
    plt.show()

def analyze_noise_effects():
    """Same IC, different noise levels."""
    net = WongWangNetwork()
    noise_levels = [0.0, 0.02, 0.05, 0.08]
    coh = 0

    fig, axes = plt.subplots(2, len(noise_levels), figsize=(18, 8))
    for i, noise in enumerate(noise_levels):
        outcomes = []
        ax_traj = axes[0, i]
        ax_bar  = axes[1, i]

        for _ in range(12):
            ic = [0.25, 0.25, 0.1]
            t = np.linspace(0, 1500, 300)
            try:
                sol = odeint(wong_wang_dynamics, ic, t, args=(coh, net, noise), mxstep=5000)
                s1, s2, _ = sol.T
            except Exception:
                # treat failure as a tie (rare)
                outcomes.append(0)
                continue

            _safe_plot(ax_traj, s1, s2, alpha=0.6, lw=1.8)

            diff = (s1[-1] - s2[-1]) if np.isfinite(s1[-1]) and np.isfinite(s2[-1]) else 0.0
            if diff > 0.1:
                outcomes.append(1)      # pop 1
            elif diff < -0.1:
                outcomes.append(2)      # pop 2
            else:
                outcomes.append(0)      # tie

        # bar chart
        counts = [outcomes.count(1), outcomes.count(0), outcomes.count(2)]
        ax_bar.bar(['pop1', 'tie', 'pop2'], counts, color=['red', 'gray', 'blue'], edgecolor='black')
        ax_traj.set_title(f"σ={noise:g}")
        ax_bar.set_title(f"σ={noise:g}")
        ax_traj.set_xlim(0, 0.8); ax_traj.set_ylim(0, 0.8)
        ax_bar.set_ylim(0, max(counts) + 1)
        ax_traj.grid(True, alpha=0.3); ax_bar.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def demonstrate_hysteresis():
    """Ramp coherence up and down."""
    net = WongWangNetwork()
    coh_up   = np.linspace(0, 25, 15)
    coh_down = np.linspace(25, 0, 15)

    state = [0.2, 0.2, 0.1]
    states_up, states_down = [], []

    for coh in coh_up:
        try:
            sol = odeint(wong_wang_dynamics, state, np.linspace(0, 600, 150), args=(coh, net, 0.01), mxstep=5000)
            state = _clip01(sol[-1])
            states_up.append(state)
        except Exception:
            states_up.append(state)  # just carry forward

    for coh in coh_down:
        try:
            sol = odeint(wong_wang_dynamics, state, np.linspace(0, 600, 150), args=(coh, net, 0.01), mxstep=5000)
            state = _clip01(sol[-1])
            states_down.append(state)
        except Exception:
            states_down.append(state)

    s1_up = [s[0] for s in states_up];   s2_up = [s[1] for s in states_up]
    s1_dn = [s[0] for s in states_down]; s2_dn = [s[1] for s in states_down]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # difference vs coherence (use same coherence arrays’ lengths)
    diff_up = np.array(s1_up) - np.array(s2_up)
    diff_dn = np.array(s1_dn) - np.array(s2_dn)
    ax[0].plot(coh_up,   diff_up, 'ro-', lw=2, ms=5, label='coh ↑', alpha=0.85)
    ax[0].plot(coh_down, diff_dn, 'bo-', lw=2, ms=5, label='coh ↓', alpha=0.85)
    ax[0].axhline(0, color='k', ls='--', alpha=0.5)
    ax[0].set_xlabel('Motion coherence (%)')
    ax[0].set_ylabel('s1 - s2')
    ax[0].set_title('Hysteresis (difference)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # state space path
    _safe_plot(ax[1], s1_up, s2_up,  color='red',  lw=2, marker='o', label='coh ↑', alpha=0.85)
    _safe_plot(ax[1], s1_dn, s2_dn,  color='blue', lw=2, marker='o', label='coh ↓', alpha=0.85)
    ax[1].set_xlabel('s1'); ax[1].set_ylabel('s2'); ax[1].set_title('State path')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_phase_space()
    demonstrate_bistability()
    analyze_noise_effects()
    demonstrate_hysteresis()
    print("\nDone with Exercise 5")