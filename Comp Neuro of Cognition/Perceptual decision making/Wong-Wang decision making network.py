# ------------------------------------------------------------
# Exercise 1 (what I'm doing here)
# ------------------------------------------------------------
# Goal: get a feel for the Wong–Wang decision network
# diving into long simulations.
# 1) visualize_network_structure():
#    - draws a simple schematic of the two excitatory pops (E1/E2)
#      and one inhibitory pop (I), plus the main connections.
#    - shows synaptic time constants (AMPA/NMDA/GABA).
#    - shows a basic neuron f–I curve.
#    - quick cartoon of motion coherence levels.
#    - small table of parameters and a compact equations box.
# 2) demonstrate_basic_dynamics():
#    - runs a short, no-stimulus Euler step (500 ms).
#    - plots s1/s2, rates, difference, and a phase path.
#    - prints which side (if any) the network drifts toward.
#
# Reference: Wong & Wang (2006) J. Neuroscience
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

print("="*60)
print("Exercise 1: network structure & basic dynamics")
print("="*60)

class WongWangNetwork:
    """Model parameters for the Wong–Wang decision network"""

    def __init__(self):
        # Population sizes
        self.N_E1 = 240      # Selective excitatory population 1
        self.N_E2 = 240      # Selective excitatory population 2
        self.N_I  = 60       # Inhibitory population

        # Synaptic weights
        self.w_plus  = 1.7   # Recurrent excitation (within population)
        self.w_minus = 1.0   # Cross-inhibition (between populations)
        self.w_I     = 1.0   # Global inhibition strength

        # Time constants (ms)
        self.tau_s   = 100.0   # Synaptic gating variable
        self.tau_NMDA = 100.0  # NMDA receptors
        self.tau_AMPA = 2.0    # AMPA receptors
        self.tau_GABA = 10.0   # GABA receptors

        # Input parameters
        self.I0    = 0.3255     # Background input
        self.JA_ext = 0.00052   # External input coupling
        self.mu0   = 40.0       # Base input rate (Hz)

        # Nonlinearity / noise
        self.gamma = 0.641       # Kinetic parameter
        self.sigma = 0.02        # Noise strength

        print("Params: E1/E2/I =", self.N_E1, self.N_E2, self.N_I,
              "| w+ =", self.w_plus, "w- =", self.w_minus, "tau_s =", self.tau_s, "ms")

def visualize_network_structure():
    """Static plots that summarize the model ingredients"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Wong–Wang network overview', fontsize=15)

    network = WongWangNetwork()

    # --- Plot 1: Network schematic
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    pop1 = FancyBboxPatch((1, 5), 2.5, 1.5, boxstyle="round,pad=0.1",
                          facecolor='lightcoral', edgecolor='darkred', linewidth=2)
    ax.add_patch(pop1)
    ax.text(2.25, 5.75, 'Population 1\n240 neurons', ha='center', va='center', fontsize=10)

    pop2 = FancyBboxPatch((6.5, 5), 2.5, 1.5, boxstyle="round,pad=0.1",
                          facecolor='lightblue', edgecolor='darkblue', linewidth=2)
    ax.add_patch(pop2)
    ax.text(7.75, 5.75, 'Population 2\n240 neurons', ha='center', va='center', fontsize=10)

    pop_I = FancyBboxPatch((4, 2), 2, 1.2, boxstyle="round,pad=0.1",
                           facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(pop_I)
    ax.text(5, 2.6, 'Inhibitory\n60 neurons', ha='center', va='center', fontsize=10)

    # self-excitation
    ax.annotate('', xy=(1.5, 6.8), xytext=(2.5, 6.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(2, 7.2, f'w+ = {network.w_plus}', ha='center', color='red')

    ax.annotate('', xy=(8.5, 6.8), xytext=(7.5, 6.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax.text(8, 7.2, f'w+ = {network.w_plus}', ha='center', color='blue')

    # cross-inhibition
    ax.annotate('', xy=(6.2, 5.75), xytext=(3.8, 5.75),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple', linestyle='--'))
    ax.text(5, 6.2, f'w- = {network.w_minus}', ha='center', color='purple')

    # global inhibition
    ax.annotate('', xy=(3.5, 4.8), xytext=(4.5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(6.5, 4.8), xytext=(5.5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(5, 1.5, f'global inh: w_I = {network.w_I}', ha='center', color='gray')

    # external inputs (cartoon)
    ax.annotate('', xy=(2.25, 4.8), xytext=(2.25, 4),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(2.25, 3.5, 'input 1', ha='center', color='green')

    ax.annotate('', xy=(7.75, 4.8), xytext=(7.75, 4),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(7.75, 3.5, 'input 2', ha='center', color='green')

    ax.set_title('Schematic')
    ax.axis('off')

    # --- Plot 2: Synaptic time constants
    ax = axes[0, 1]
    t = np.linspace(0, 400, 1000)
    ampa_decay = np.exp(-t / network.tau_AMPA)
    nmda_decay = np.exp(-t / network.tau_NMDA)
    gaba_decay = np.exp(-t / network.tau_GABA)
    ax.plot(t, ampa_decay, 'r-', linewidth=3, label=f'AMPA (τ={network.tau_AMPA} ms)')
    ax.plot(t, nmda_decay, 'b-', linewidth=3, label=f'NMDA (τ={network.tau_NMDA} ms)')
    ax.plot(t, gaba_decay, 'g-', linewidth=3, label=f'GABA (τ={network.tau_GABA} ms)')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('normalized')
    ax.set_title('Synaptic time courses')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 400)

    # --- Plot 3: simple f–I curve
    ax = axes[0, 2]
    I_input = np.linspace(-0.5, 2, 200)
    firing_rate = np.maximum(0, I_input * 100)
    firing_rate = np.minimum(firing_rate, 80)
    ax.plot(I_input, firing_rate, 'k-', linewidth=3)
    ax.axvline(x=network.I0, color='r', linestyle='--', linewidth=2, label=f'I0 = {network.I0}')
    ax.set_xlabel('input current (arb.)')
    ax.set_ylabel('rate (Hz)')
    ax.set_title('f–I (toy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(0, 85)

    # --- Plot 4: coherence cartoon
    ax = axes[1, 0]
    coherence_levels = [0, 6.4, 12.8, 25.6, 51.2]
    colors = ['black', 'darkblue', 'blue', 'orange', 'red']
    for i, (coh, color) in enumerate(zip(coherence_levels, colors)):
        y_pos = 4.5 - i * 0.8
        noise_x = np.random.uniform(0.5, 4.5, 20)
        noise_y = np.random.uniform(y_pos - 0.3, y_pos + 0.3, 20)
        ax.scatter(noise_x, noise_y, s=15, c='lightgray', alpha=0.6)
        n_coherent = int(coh / 100 * 10) + 1
        coherent_x = np.linspace(1, 4, n_coherent)
        coherent_y = np.full(n_coherent, y_pos)
        for x, y in zip(coherent_x, coherent_y):
            ax.annotate('', xy=(x + 0.3, y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color=color))
        ax.text(5, y_pos, f'{coh}%', va='center', color=color, fontsize=11)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.set_title('Coherence levels (cartoon)')
    ax.axis('off')

    # --- Plot 5: parameter table
    ax = axes[1, 1]
    param_data = [
        ['Parameter', 'Value', 'Description'],
        ['N_E1, N_E2', f'{network.N_E1}', 'Selective populations'],
        ['N_I', f'{network.N_I}', 'Inhibitory population'],
        ['w+', f'{network.w_plus}', 'Self-excitation'],
        ['w-', f'{network.w_minus}', 'Cross-inhibition'],
        ['w_I', f'{network.w_I}', 'Global inhibition'],
        ['τ_s', f'{network.tau_s} ms', 'Synaptic time constant'],
        ['τ_NMDA', f'{network.tau_NMDA} ms', 'NMDA decay'],
        ['I0', f'{network.I0}', 'Background input'],
        ['γ', f'{network.gamma}', 'Kinetic parameter'],
        ['σ', f'{network.sigma}', 'Noise strength']
    ]
    table = ax.table(cellText=param_data[1:], colLabels=param_data[0],
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Parameters')
    ax.axis('off')

    # --- Plot 6: compact equations box
    ax = axes[1, 2]
    equations_text = (
        "ds1/dt = (-s1 + γ r1 (1-s1)) / τs\n"
        "ds2/dt = (-s2 + γ r2 (1-s2)) / τs\n\n"
        "I1 = I0 + JA·ν1 + w+ s1 - w- s2 - wI sI\n"
        "I2 = I0 + JA·ν2 + w+ s2 - w- s1 - wI sI\n\n"
        "r = max(0, I)\n"
        "ν1 = μ0 (1 + c/100),  ν2 = μ0 (1 - c/100)\n"
        "decision: choose pop with rate above threshold"
    )
    ax.text(0.05, 0.95, equations_text, transform=ax.transAxes,
            fontsize=11, va='top', family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.set_title('Equations (condensed)')
    ax.axis('off')

    plt.tight_layout()
    plt.show()
    return network

def demonstrate_basic_dynamics():
    """Short no-stimulus run to see how s1/s2 evolve"""
    print("\nRunning short no-stimulus dynamics (Euler, 500 ms)")

    network = WongWangNetwork()

    # time grid
    dt = 1.0          # ms
    t_total = 500     # ms
    time = np.arange(0, t_total, dt)

    # init
    s1 = np.zeros_like(time)
    s2 = np.zeros_like(time)
    s1[0] = 0.1
    s2[0] = 0.1

    # Euler integration
    for i in range(1, len(time)):
        I1 = network.I0 + network.w_plus * s1[i-1] - network.w_minus * s2[i-1]
        I2 = network.I0 + network.w_plus * s2[i-1] - network.w_minus * s1[i-1]
        r1 = max(0, I1)
        r2 = max(0, I2)
        ds1_dt = (-s1[i-1] + network.gamma * r1 * (1 - s1[i-1])) / network.tau_s
        ds2_dt = (-s2[i-1] + network.gamma * r2 * (1 - s2[i-1])) / network.tau_s
        s1[i] = s1[i-1] + dt * ds1_dt + 0.001 * np.random.randn()
        s2[i] = s2[i-1] + dt * ds2_dt + 0.001 * np.random.randn()
        s1[i] = max(0, min(1, s1[i]))
        s2[i] = max(0, min(1, s2[i]))

    # simple rate proxy for plotting
    r1 = np.maximum(0, network.I0 + network.w_plus * s1 - network.w_minus * s2) * 100
    r2 = np.maximum(0, network.I0 + network.w_plus * s2 - network.w_minus * s1) * 100

    # plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Basic dynamics (no stimulus)', fontsize=15)

    axes[0, 0].plot(time, s1, 'r-', linewidth=3, label='s1')
    axes[0, 0].plot(time, s2, 'b-', linewidth=3, label='s2')
    axes[0, 0].set_xlabel('time (ms)')
    axes[0, 0].set_ylabel('s')
    axes[0, 0].set_title('Synaptic variables')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time, r1, 'r-', linewidth=3, label='r1')
    axes[0, 1].plot(time, r2, 'b-', linewidth=3, label='r2')
    axes[0, 1].set_xlabel('time (ms)')
    axes[0, 1].set_ylabel('Hz (proxy)')
    axes[0, 1].set_title('Rates (proxy)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time, s1 - s2, 'purple', linewidth=3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('time (ms)')
    axes[1, 0].set_ylabel('s1 - s2')
    axes[1, 0].set_title('Competition (difference)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(s1, s2, 'g-', linewidth=2, alpha=0.8)
    axes[1, 1].plot(s1[0], s2[0], 'go', markersize=8, label='start')
    axes[1, 1].plot(s1[-1], s2[-1], 'ro', markersize=8, label='end')
    axes[1, 1].set_xlabel('s1')
    axes[1, 1].set_ylabel('s2')
    axes[1, 1].set_title('Phase path')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, max(0.5, max(s1) * 1.1))
    axes[1, 1].set_ylim(0, max(0.5, max(s2) * 1.1))

    plt.tight_layout()
    plt.show()

    # quick text summary
    final_s1 = s1[-1]
    final_s2 = s2[-1]
    diff = abs(final_s1 - final_s2)
    print(f"final s1={final_s1:.3f}, s2={final_s2:.3f}, |Δ|={diff:.3f}")
    if diff < 0.1:
        print("balance (no clear winner)")
    elif final_s1 > final_s2:
        print("drift toward pop 1")
    else:
        print("drift toward pop 2")

if __name__ == "__main__":
    print("Starting Exercise 1")
    _ = visualize_network_structure()
    demonstrate_basic_dynamics()
    print("="*60)
    print("Exercise 1 done")
    print("="*60)