"""
Spatial Working Memory model

This script builds a simple "ring network" to explore spatial working memory.
- Neurons are arranged in a circle, each with a preferred angle (0–360°).
- Nearby neurons excite each other, while distant ones inhibit each other 
  (a "Mexican hat" connectivity profile).
- A cue activates neurons around one angle (like 90°), creating a "bump"
  of activity. After the cue ends, that bump can persist = working memory.
- We’ll test how this looks with a single cue, then multiple cues to see
  how much information the network can hold at once.

Figures show the structure, weights, activity patterns, and performance.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("SPATIAL WORKING MEMORY MODEL")
print("="*60)

# Try loading neurodynex modules (not required for the custom model below)
try:
    import neurodynex3.working_memory_network.wm_model as wm_model
    print("✓ Imported wm_model")
    functions = [f for f in dir(wm_model) if not f.startswith('_')]
    print("Available functions:", functions)
except ImportError:
    print("⚠️ wm_model not available")

try:
    import neurodynex3.tools.plot_tools as plot_tools
    print("✓ Imported plot_tools")
except ImportError:
    print("⚠️ plot_tools not available")


# -----------------------------------------------------------------------------
# Build the ring network
# -----------------------------------------------------------------------------
def create_ring_network(N=50):
    """
    Make a ring of N neurons with Mexican hat connectivity.
    Each neuron has a preferred direction (0–360°).
    """
    tau_e = 20.0  # excitatory time constant (ms)
    tau_i = 10.0  # inhibitory time constant (ms)

    # Preferred directions spread evenly around the circle
    preferred_dirs = np.linspace(0, 360, N, endpoint=False)

    # Build weight matrix
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            angle_diff = abs(preferred_dirs[i] - preferred_dirs[j])
            angle_diff = min(angle_diff, 360 - angle_diff)  # wrap-around
            if angle_diff <= 45:
                W[i, j] = 0.5 * np.exp(-angle_diff**2 / (2 * 20**2))  # local excitation
            else:
                W[i, j] = -0.1  # global inhibition
    np.fill_diagonal(W, 0)

    return {
        "N": N,
        "preferred_dirs": preferred_dirs,
        "weights": W,
        "tau_e": tau_e,
        "tau_i": tau_i
    }


# -----------------------------------------------------------------------------
# Visualize the ring and connectivity
# -----------------------------------------------------------------------------
def show_ring_network(network):
    N = network["N"]
    dirs = network["preferred_dirs"]
    W = network["weights"]

    plt.figure(figsize=(15, 10))

    # 1. Ring layout
    plt.subplot(2, 3, 1)
    angles = np.radians(dirs)
    plt.scatter(np.cos(angles), np.sin(angles), c=dirs, cmap="hsv", s=40)
    plt.title("Ring layout (preferred directions)")
    plt.axis("equal")

    # 2. Full weight matrix
    plt.subplot(2, 3, 2)
    plt.imshow(W, cmap="RdBu_r", aspect="auto")
    plt.colorbar(label="Weight")
    plt.title("Connectivity matrix")

    # 3. Profile of one neuron
    plt.subplot(2, 3, 3)
    idx = 12  # neuron ~90°
    plt.plot(dirs, W[idx, :], "b-", lw=2)
    plt.axvline(dirs[idx], color="r", ls="--", label=f"Neuron {dirs[idx]:.0f}°")
    plt.title("Connectivity profile")
    plt.xlabel("Presynaptic direction (°)")
    plt.ylabel("Weight")
    plt.legend()

    # 4. Profiles of a few neurons
    plt.subplot(2, 3, 4)
    for i in [0, 12, 25, 37]:
        plt.plot(dirs, W[i, :], lw=2, label=f"{dirs[i]:.0f}°")
    plt.title("Profiles for multiple neurons")
    plt.xlabel("Presynaptic direction (°)")
    plt.ylabel("Weight")
    plt.legend()

    # 5. Weight histogram
    plt.subplot(2, 3, 5)
    plt.hist(W[W > 0], bins=20, color="red", alpha=0.7, label="Excitatory")
    plt.hist(W[W < 0], bins=20, color="blue", alpha=0.7, label="Inhibitory")
    plt.title("Weight distribution")
    plt.legend()

    # 6. Network summary
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, f"N = {N}", fontsize=11)
    plt.text(0.1, 0.8, f"τ_e = {network['tau_e']} ms", fontsize=11)
    plt.text(0.1, 0.7, f"τ_i = {network['tau_i']} ms", fontsize=11)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Run a simple working memory simulation
# -----------------------------------------------------------------------------
def simulate(network, cue_dir=90, cue_strength=2.0,
             t_cue_start=100, t_cue_end=200, t_total=800, dt=1.0):
    """
    Run network dynamics with one cue.
    Returns time array and firing rates (neurons x time).
    """
    N = network["N"]
    dirs = network["preferred_dirs"]
    W = network["weights"]
    tau = network["tau_e"]

    n_steps = int(t_total / dt)
    time = np.arange(0, t_total, dt)
    rates = np.zeros((N, n_steps))
    rates[:, 0] = 0.1  # small baseline

    for t_idx in range(1, n_steps):
        t = time[t_idx]
        ext = np.zeros(N)
        if t_cue_start <= t <= t_cue_end:
            for i in range(N):
                angle_diff = abs(dirs[i] - cue_dir)
                angle_diff = min(angle_diff, 360 - angle_diff)
                ext[i] = cue_strength * np.exp(-angle_diff**2 / (2 * 15**2))

        net_input = np.dot(W, rates[:, t_idx-1]) + ext
        net_input += 0.1 * np.random.randn(N)  # noise

        rates[:, t_idx] = rates[:, t_idx-1] + dt/tau * (
            -rates[:, t_idx-1] + np.maximum(0, net_input)
        )

    return time, rates


# -----------------------------------------------------------------------------
# Show results of single cue
# -----------------------------------------------------------------------------
def show_single_cue(network, time, rates, target_dir=90):
    dirs = network["preferred_dirs"]

    plt.figure(figsize=(16, 10))

    # 1. Heatmap of activity over time
    plt.subplot(2, 3, 1)
    plt.imshow(rates, aspect="auto", cmap="hot", origin="lower",
               extent=[0, time[-1], 0, 360])
    plt.axvline(100, color="cyan", ls="--")
    plt.axvline(200, color="blue", ls="--")
    plt.title("Activity evolution")

    # 2. Profiles at a few time points
    plt.subplot(2, 3, 2)
    for t_point, c, label in zip([50, 150, 300, 600],
                                 ["blue", "red", "green", "purple"],
                                 ["Pre", "Cue", "Early mem", "Late mem"]):
        idx = int(t_point)
        plt.plot(dirs, rates[:, idx], color=c, lw=2, label=f"{label} ({t_point}ms)")
    plt.axvline(target_dir, color="k", ls="--")
    plt.legend()
    plt.title("Profiles at different times")

    # 3. Peak location over time
    plt.subplot(2, 3, 3)
    peaks = [dirs[np.argmax(rates[:, i])] if np.max(rates[:, i]) > 0.5 else np.nan
             for i in range(len(time))]
    plt.plot(time, peaks, "r-")
    plt.axhline(target_dir, color="k", ls="--")
    plt.title("Peak location over time")

    # 4. Peak amplitude
    plt.subplot(2, 3, 4)
    amps = [np.max(rates[:, i]) for i in range(len(time))]
    plt.plot(time, amps, "g-")
    plt.axvline(100, color="cyan", ls="--")
    plt.axvline(200, color="blue", ls="--")
    plt.title("Peak amplitude")

    # 5. Total network activity
    plt.subplot(2, 3, 5)
    plt.plot(time, np.sum(rates, axis=0), "k-")
    plt.title("Total activity")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Multiple cues test
# -----------------------------------------------------------------------------
def multi_cue_test(network):
    configs = [
        {"name": "Single cue", "dirs": [90]},
        {"name": "Two cues (90° apart)", "dirs": [45, 135]},
        {"name": "Two cues (60° apart)", "dirs": [60, 120]},
        {"name": "Three cues", "dirs": [0, 120, 240]},
    ]

    plt.figure(figsize=(16, 12))

    for i, cfg in enumerate(configs):
        N = network["N"]
        dirs = network["preferred_dirs"]
        W = network["weights"]

        time = np.arange(0, 600, 1.0)
        rates = np.zeros((N, len(time)))
        rates[:, 0] = 0.1

        for t_idx in range(1, len(time)):
            t = time[t_idx]
            ext = np.zeros(N)
            if 100 <= t <= 200:
                for cue in cfg["dirs"]:
                    for j in range(N):
                        angle_diff = abs(dirs[j] - cue)
                        angle_diff = min(angle_diff, 360 - angle_diff)
                        ext[j] += 2.0 * np.exp(-angle_diff**2 / (2 * 15**2))

            net_input = np.dot(W, rates[:, t_idx-1]) + ext
            net_input += 0.1 * np.random.randn(N)
            rates[:, t_idx] = rates[:, t_idx-1] + 1.0/20.0 * (
                -rates[:, t_idx-1] + np.maximum(0, net_input)
            )

        # Plot heatmap
        plt.subplot(2, 4, i+1)
        plt.imshow(rates, aspect="auto", cmap="hot", origin="lower",
                   extent=[0, 600, 0, 360])
        for d in cfg["dirs"]:
            plt.axhline(d, color="cyan", ls="--")
        plt.title(f"{cfg['name']}")

        # Final activity profile
        plt.subplot(2, 4, i+5)
        final_rates = np.mean(rates[:, -100:], axis=1)
        plt.plot(dirs, final_rates, "b-", lw=2)
        for d in cfg["dirs"]:
            plt.axvline(d, color="r", ls="--")
        plt.title("Final profile")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    net = create_ring_network(50)
    show_ring_network(net)

    time, rates = simulate(net, cue_dir=90, cue_strength=3.0)
    show_single_cue(net, time, rates)

    multi_cue_test(net)
    print("Done.")