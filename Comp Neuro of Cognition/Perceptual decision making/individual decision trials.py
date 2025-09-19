"""
Exercise 2 — Single-trial dynamics (Wong–Wang)

What I’m doing in this script (plain English):
- Run the decision circuit for one **trial** at a time.
- Turn the stimulus on/off during the trial and watch the two pools compete.
- Call a decision when one pool stays above a small threshold.
- Repeat at a few coherence levels to see RT/choice/“confidence” change.

This is meant to be readable and simple, not a perfect replica of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("=" * 60)
print("EXERCISE 2: SINGLE-TRIAL DECISION DYNAMICS")
print("=" * 60)

# small helpers
_RNG = np.random.default_rng(2)
def _clip01(x): return float(np.clip(x, 0.0, 1.0))


class WongWangNetwork:
    """Tiny parameter set for a clean winner-take-all demo."""
    def __init__(self):
        # recurrent structure
        self.w_plus  = 1.7    # self-excitation
        self.w_minus = 1.0    # cross-inhibition
        self.w_I     = 1.0    # global inhibition

        # time constant (ms)
        self.tau_s   = 100.0

        # background + stimulus coupling
        self.I0      = 0.3255
        self.JA_ext  = 0.00052
        self.mu0     = 40.0

        # gain + noise
        self.gamma   = 0.641
        self.sigma   = 0.02

        print("Network ready.")


def wong_wang_dynamics(state, t, coherence, net, stimulus_on=True):
    """
    One step of the Wong–Wang dynamics.
    state = [s1, s2, sI] are synaptic gating variables (kept in [0,1]).
    """
    s1, s2, sI = (_clip01(state[0]), _clip01(state[1]), _clip01(state[2]))

    # stimulus drive from motion coherence
    c = (coherence / 100.0) if stimulus_on else 0.0
    nu1 = net.mu0 * (1 + c)
    nu2 = net.mu0 * (1 - c)
    I_ext1 = net.JA_ext * nu1 if stimulus_on else 0.0
    I_ext2 = net.JA_ext * nu2 if stimulus_on else 0.0

    # recurrent + inhibitory
    I1 = net.I0 + I_ext1 + net.w_plus*s1 - net.w_minus*s2 - net.w_I*sI
    I2 = net.I0 + I_ext2 + net.w_plus*s2 - net.w_minus*s1 - net.w_I*sI
    I_I = net.I0 + net.gamma * (s1 + s2)

    # rectified “rates” (kept simple)
    r1 = max(0.0, I1)
    r2 = max(0.0, I2)
    rI = max(0.0, I_I)

    # synaptic updates
    ds1 = (-s1 + net.gamma * r1 * (1 - s1)) / net.tau_s
    ds2 = (-s2 + net.gamma * r2 * (1 - s2)) / net.tau_s
    dsI = (-sI + rI) / net.tau_s

    # small noise on the derivatives
    if net.sigma > 0:
        ds1 += net.sigma * _RNG.normal()
        ds2 += net.sigma * _RNG.normal()
        dsI += net.sigma * _RNG.normal()

    return [ds1, ds2, dsI]


def simulate_single_trial(net, coherence, trial_duration=2000,
                          stimulus_start=100, stimulus_end=2000, dt=2.0):
    time = np.arange(0, trial_duration, dt)
    x0 = [0.1, 0.1, 0.1]

    def dyn(state, t):
        stim_on = (stimulus_start <= t <= stimulus_end)
        return wong_wang_dynamics(state, t, coherence, net, stim_on)

    sol = odeint(dyn, x0, time, mxstep=5000)
    s1, s2, sI = sol.T

    # crude “firing rate” readout for the plots/threshold
    r1 = np.maximum(0.0, net.I0 + net.w_plus*s1 - net.w_minus*s2) * 150.0
    r2 = np.maximum(0.0, net.I0 + net.w_plus*s2 - net.w_minus*s1) * 150.0

    # decision rule
    threshold = 15.0  # Hz
    choice, decision_time = None, trial_duration
    for i, t in enumerate(time[int(stimulus_start):], start=int(stimulus_start)):
        if r1[i] > threshold and r1[i] > r2[i] + 5:
            choice, decision_time = 1, time[i]; break
        if r2[i] > threshold and r2[i] > r1[i] + 5:
            choice, decision_time = 2, time[i]; break
    if choice is None:
        choice = 1 if np.mean(r1[-100:]) >= np.mean(r2[-100:]) else 2

    # correctness (positive coherence means pop 1 is “right”)
    correct_choice = 1 if coherence > 0 else 2
    is_correct = (choice == correct_choice) if coherence != 0 else True

    return {
        'time': time,
        's1': s1, 's2': s2, 's_I': sI,
        'r1': r1, 'r2': r2,
        'decision_time': decision_time,
        'choice': choice,
        'correct': is_correct,
        'coherence': coherence,
        'stimulus_start': stimulus_start,
        'stimulus_end': stimulus_end
    }


def visualize_single_trial():
    """Run one trial per coherence and plot the main pieces."""
    print("\n" + "="*50)
    print("SINGLE TRIAL SIMULATION")
    print("="*50)

    net = WongWangNetwork()
    coherences = [0, 12.8, 25.6, 51.2]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('Single-trial decision dynamics', fontsize=16)

    results = []
    for row, c in enumerate(coherences):
        res = simulate_single_trial(net, c, trial_duration=1500,
                                    stimulus_start=100, stimulus_end=1500)
        results.append(res)

        t  = res['time']
        s1, s2 = res['s1'], res['s2']
        r1, r2 = res['r1'], res['r2']

        print(f"coh {c:>5.1f}%  choice=Pop{res['choice']}  t={res['decision_time']:.0f} ms  correct={res['correct']}")

        # (1) synaptic variables
        ax = axes[row, 0]
        ax.plot(t, s1, 'r', lw=2, alpha=0.9, label='pop 1')
        ax.plot(t, s2, 'b', lw=2, alpha=0.9, label='pop 2')
        ax.axvspan(res['stimulus_start'], res['stimulus_end'], color='gold', alpha=0.2)
        ax.axvline(res['decision_time'], color='k', ls='--', alpha=0.7)
        ax.set_ylabel('s (synaptic)'); ax.set_xlim(0, 1500); ax.grid(True, alpha=0.3)
        if row == 0: ax.legend()
        ax.set_title(f'coh {c}%')

        # (2) firing-rate proxies
        ax = axes[row, 1]
        ax.plot(t, r1, 'r', lw=2)
        ax.plot(t, r2, 'b', lw=2)
        ax.axhline(15, color='k', ls=':', alpha=0.7)
        ax.axvspan(res['stimulus_start'], res['stimulus_end'], color='gold', alpha=0.2)
        ax.axvline(res['decision_time'], color='k', ls='--', alpha=0.7)
        ax.set_ylabel('rate (Hz)'); ax.set_xlim(0, 1500); ax.set_ylim(0, 30); ax.grid(True, alpha=0.3)
        if row == 0: ax.legend(['pop 1','pop 2','threshold'])

        # (3) competition signal (difference)
        ax = axes[row, 2]
        ax.plot(t, r1 - r2, color='purple', lw=2)
        ax.axhline(0, color='k', alpha=0.4)
        ax.axvspan(res['stimulus_start'], res['stimulus_end'], color='gold', alpha=0.2)
        ax.axvline(res['decision_time'], color='k', ls='--', alpha=0.7)
        ax.set_ylabel('pop1 − pop2 (Hz)'); ax.set_xlim(0, 1500); ax.grid(True, alpha=0.3)

        # (4) phase portrait
        ax = axes[row, 3]
        ax.plot(s1, s2, 'g-', lw=2, alpha=0.85)
        ax.plot(s1[0], s2[0], 'go', ms=8)
        ax.plot(s1[-1], s2[-1], 'ro', ms=8)
        ax.set_xlabel('s1'); ax.set_ylabel('s2'); ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8); ax.grid(True, alpha=0.3)

    for col in range(4):
        axes[-1, col].set_xlabel('time (ms)')

    plt.tight_layout()
    plt.show()
    return results


def analyze_decision_process():
    """Zoom in on one moderate-coherence trial and annotate a few signals."""
    print("\n" + "="*50)
    print("DECISION PROCESS (one trial, more views)")
    print("="*50)

    net = WongWangNetwork()
    coh = 25.6
    res = simulate_single_trial(net, coh, trial_duration=2000,
                                stimulus_start=200, stimulus_end=2000)

    t = res['time']; r1, r2 = res['r1'], res['r2']
    s1, s2 = res['s1'], res['s2']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed single-trial views', fontsize=16)

    # full trace
    ax = axes[0,0]
    ax.plot(t, r1, 'r', lw=2, label='pop 1 (preferred)')
    ax.plot(t, r2, 'b', lw=2, label='pop 2')
    ax.axhline(15, color='k', ls=':', alpha=0.7)
    ax.axvspan(200, 2000, color='gold', alpha=0.2)
    ax.axvline(res['decision_time'], color='k', ls='--', alpha=0.7)
    ax.set_xlim(0, 2000); ax.grid(True, alpha=0.3); ax.legend(); ax.set_title('whole trial')

    # zoom on buildup
    ax = axes[0,1]
    mask = (t >= 150) & (t <= 800)
    ax.plot(t[mask], r1[mask], 'r', lw=2)
    ax.plot(t[mask], r2[mask], 'b', lw=2)
    ax.axhline(15, color='k', ls=':', alpha=0.7)
    ax.axvspan(200, 800, color='gold', alpha=0.2)
    ax.grid(True, alpha=0.3); ax.set_title('buildup')

    # bias ratio
    ax = axes[0,2]
    ratio = r1 / (r1 + r2 + 1e-6)
    ax.plot(t, ratio, color='purple', lw=2)
    ax.axhline(0.5, color='k', alpha=0.4)
    ax.axvspan(200, 2000, color='gold', alpha=0.2)
    ax.axvline(res['decision_time'], color='k', ls='--', alpha=0.7)
    ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.set_title('bias ratio')

    # confidence proxy
    ax = axes[1,0]
    conf = np.abs(r1 - r2)
    ax.plot(t, conf, color='orange', lw=2)
    ax.axvspan(200, 2000, color='gold', alpha=0.2)
    ax.axvline(res['decision_time'], color='k', ls='--', alpha=0.7)
    ax.grid(True, alpha=0.3); ax.set_title('|pop1 − pop2|')

    # state trajectory (colored by time)
    ax = axes[1,1]
    dots = ax.scatter(s1[::10], s2[::10], c=t[::10], cmap='viridis', s=30, alpha=0.8)
    ax.plot(s1[0], s2[0], 'go', ms=10); ax.plot(s1[-1], s2[-1], 'ro', ms=10)
    plt.colorbar(dots, ax=ax, label='time (ms)')
    ax.set_xlim(0,0.8); ax.set_ylim(0,0.8); ax.grid(True, alpha=0.3); ax.set_title('phase (colored)')

    # quick numbers
    ax = axes[1,2]
    pre  = t < 200
    post = ~pre
    txt = f"""Trial stats (coh {coh}%)
choice: Pop {res['choice']}   RT: {res['decision_time']-200:.0f} ms   correct: {res['correct']}
baseline (pre): pop1 {np.mean(r1[pre]):.1f} Hz, pop2 {np.mean(r2[pre]):.1f} Hz
peaks (post):   pop1 {np.max(r1[post]):.1f} Hz, pop2 {np.max(r2[post]):.1f} Hz
final 100 ms:   Δ={abs(np.mean(r1[-100:]) - np.mean(r2[-100:])):.1f} Hz
"""
    ax.text(0.03, 0.97, txt, va='top', family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"done. RT = {res['decision_time']-200:.0f} ms  choice=Pop{res['choice']}  correct={res['correct']}")


def compare_coherence_effects():
    """Run one trial at several coherences and summarise."""
    print("\n" + "="*50)
    print("COHERENCE COMPARISON")
    print("="*50)

    net = WongWangNetwork()
    levels = [0, 6.4, 12.8, 25.6, 51.2]

    results = []
    for c in levels:
        res = simulate_single_trial(net, c, trial_duration=1500)
        results.append(res)
        print(f"  {c:>5.1f}% → RT {res['decision_time']-100:.0f} ms, choice Pop{res['choice']}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Effect of coherence (one trial per level)', fontsize=16)

    # (1) difference traces
    ax = axes[0,0]
    colors = ['black','tab:blue','tab:green','tab:orange','tab:red']
    for res, col in zip(results, colors):
        ax.plot(res['time'], res['r1']-res['r2'], color=col, lw=2, alpha=0.85,
                label=f"{res['coherence']}%")
        ax.axvline(res['decision_time'], color=col, ls='--', alpha=0.5)
    ax.axhline(0, color='k', alpha=0.3)
    ax.axvspan(100, 1500, color='gold', alpha=0.2)
    ax.set_xlim(0,1500); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('pop1 − pop2 (Hz)')

    # (2) reaction times
    ax = axes[0,1]
    rts = [res['decision_time'] - 100 for res in results]
    bars = ax.bar(range(len(levels)), rts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f'{c}%' for c in levels])
    ax.set_ylabel('RT (ms)'); ax.grid(True, axis='y', alpha=0.3); ax.set_title('RT vs coherence')
    for b, rt in zip(bars, rts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+8, f'{rt:.0f}', ha='center', va='bottom')

    # (3) final activities
    ax = axes[0,2]
    final_r1 = [np.mean(r['r1'][-50:]) for r in results]
    final_r2 = [np.mean(r['r2'][-50:]) for r in results]
    x = np.arange(len(levels)); w = 0.42
    ax.bar(x-w/2, final_r1, width=w, color='red',  alpha=0.75, label='pop 1', edgecolor='black')
    ax.bar(x+w/2, final_r2, width=w, color='blue', alpha=0.75, label='pop 2', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([f'{c}%' for c in levels])
    ax.set_ylabel('final activity (Hz)'); ax.grid(True, axis='y', alpha=0.3); ax.legend()
    ax.set_title('final 50 ms')

    # (4) confidence proxy
    ax = axes[1,0]
    conf = [abs(np.mean(r['r1'][-50:]) - np.mean(r['r2'][-50:])) for r in results]
    ax.bar(range(len(levels)), conf, color='purple', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f'{c}%' for c in levels])
    ax.set_ylabel('|pop1 − pop2| (Hz)'); ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('confidence (rough)')

    # (5) choices (circle = correct, X = error)
    ax = axes[1,1]
    for c, res in zip(levels, results):
        marker = 'o' if res['correct'] else 'X'
        color  = 'red' if res['choice']==1 else 'blue'
        ax.scatter(c, res['choice'], s=180, c=color, marker=marker, edgecolor='black', linewidth=1.5)
    ax.set_yticks([1,2]); ax.set_ylim(0.5, 2.5)
    ax.set_xlabel('coherence (%)'); ax.set_ylabel('choice (population)')
    ax.set_title('choices & correctness'); ax.grid(True, alpha=0.3)

    # (6) quick text panel
    ax = axes[1,2]
    txt = f"""Summary
RTs (ms): {', '.join(f'{int(v)}' for v in rts)}
RT speedup (0% → {levels[-1]}%): {int(rts[0]-rts[-1])} ms
Confidence range: {min(conf):.1f}–{max(conf):.1f} Hz
Takeaways:
• More coherence → faster decisions.
• More coherence → larger pop difference (confidence).
• Same circuit, behavior shifts with input strength."""
    ax.text(0.03, 0.97, txt, va='top', family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.85))
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"RT range: {min(rts):.0f}–{max(rts):.0f} ms   Confidence: {min(conf):.1f}–{max(conf):.1f} Hz")


if __name__ == "__main__":
    print("Startinçg Exercise 2 …")
    visualize_single_trial()
    analyze_decision_process()
    compare_coherence_effects()
    print("\nDone with Exercise 2.")