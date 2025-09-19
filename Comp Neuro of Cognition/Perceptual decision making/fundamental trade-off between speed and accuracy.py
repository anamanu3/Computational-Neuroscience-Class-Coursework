"""
Exercise 4 — Speed/Accuracy trade-off (Wong–Wang)

What I’m doing in this file (plain English):
- I run the same decision circuit while changing the **decision threshold**.
  Lower threshold = commit earlier (faster, sloppier). Higher = wait longer (slower, cleaner).
- I also add an **urgency** term that ramps up with time (like time pressure).
- I run a few sweeps to show:
    1) what thresholds do to example trials, RT histograms, and accuracy
    2) a small grid (threshold × coherence) to draw the classic speed–accuracy curves
    3) what different urgency schedules look like on the exact same task
Nothing fancy — just enough to see the trade-offs in action.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

print("=" * 60)
print("EXERCISE 4: SPEED–ACCURACY TRADE-OFF (simple demos)")
print("=" * 60)

# I use a tiny clamp and a fixed RNG so runs are reproducible and stable.
_RNG = np.random.default_rng(3)
def _clip01(x): return float(np.clip(x, 0.0, 1.0))


class WongWangNetwork:
    """Minimal parameter set for the attractor model (same as earlier exercises)."""
    def __init__(self):
        # synaptic & input params tuned for clean winner-take-all
        self.w_plus  = 1.8
        self.w_minus = 1.2
        self.w_I     = 1.0
        self.tau_s   = 60.0
        self.I0      = 0.3
        self.JA_ext  = 0.005
        self.mu0     = 30.0
        self.gamma   = 0.641
        self.sigma   = 0.05  # small state noise
        print("Wong–Wang network ready (threshold/urgency demos)")


def wong_wang_dynamics_with_urgency(state, t, coherence, network, threshold,
                                    urgency_onset=0, urgency_slope=0):
    """Basic Wong–Wang step + a simple urgency term that grows after `urgency_onset`."""
    s1, s2, s_I = (_clip01(state[0]), _clip01(state[1]), _clip01(state[2]))

    # stimulus drive (coherence c maps to ν1, ν2)
    c   = float(coherence) / 100.0
    nu1 = network.mu0 * (1 + c)
    nu2 = network.mu0 * (1 - c)
    I_ext1 = network.JA_ext * nu1
    I_ext2 = network.JA_ext * nu2

    # urgency (increases net drive symmetrically)
    urg = 0.0
    if t > urgency_onset:
        urg = urgency_slope * (t - urgency_onset) / 1000.0  # mild scale

    # recurrent + inhibition + urgency
    I1   = network.I0 + I_ext1 + network.w_plus*s1 - network.w_minus*s2 - network.w_I*s_I + urg
    I2   = network.I0 + I_ext2 + network.w_plus*s2 - network.w_minus*s1 - network.w_I*s_I + urg
    I_in = network.I0 + network.gamma*(s1 + s2)

    # rectified rates (just to keep it simple)
    r1 = max(0.0, I1 * 100.0)
    r2 = max(0.0, I2 * 100.0)
    rI = max(0.0, I_in * 50.0)

    # synaptic dynamics
    ds1 = (-s1 + network.gamma*(r1/100.0)*(1.0 - s1)) / network.tau_s
    ds2 = (-s2 + network.gamma*(r2/100.0)*(1.0 - s2)) / network.tau_s
    dsI = (-s_I + rI/50.0) / network.tau_s

    # small noise on the derivatives
    if network.sigma > 0:
        ds1 += network.sigma * _RNG.normal()
        ds2 += network.sigma * _RNG.normal()
        dsI += 0.5 * network.sigma * _RNG.normal()

    return [ds1, ds2, dsI]


def simulate_trial_with_threshold(network, coherence, threshold, max_time=2000,
                                  urgency_onset=0, urgency_slope=0):
    """One trial: integrate, then call a decision when a population stays over threshold."""
    dt = 0.5
    T  = np.arange(0, max_time, dt)
    x0 = [0.05, 0.05, 0.05]

    try:
        sol = odeint(
            wong_wang_dynamics_with_urgency, x0, T,
            args=(coherence, network, threshold, urgency_onset, urgency_slope),
            mxstep=5000
        )
        s1, s2, _ = sol.T
    except Exception as e:
        # rare, but just in case the solver hiccups
        print("(solver skipped a trial:", e, ")")
        s1 = s2 = np.zeros_like(T)

    # “firing rates” proxy for threshold readout
    r1 = np.maximum(0.0, (network.I0 + network.w_plus * s1 - network.w_minus * s2) * 200.0)
    r2 = np.maximum(0.0, (network.I0 + network.w_plus * s2 - network.w_minus * s1) * 200.0)

    # decision rule: sustained crossing for ~30 ms
    choice, decision_time = None, max_time
    win = int(30 / dt)
    for i in range(int(200/dt), len(T) - win):
        m1 = float(np.mean(r1[i:i+win]))
        m2 = float(np.mean(r2[i:i+win]))
        if m1 > threshold and (m1 > m2 + 5):
            choice, decision_time = 1, T[i]; break
        if m2 > threshold and (m2 > m1 + 5):
            choice, decision_time = 2, T[i]; break

    # if nobody gets there, pick the bigger at the end (counts as timeout)
    if choice is None:
        if np.mean(r1[-2*win:]) >= np.mean(r2[-2*win:]):
            choice = 1
        else:
            choice = 2
        decision_time = max_time

    # ground-truth side = population 1 if c>0, pop 2 if c<0, else coin flip
    if coherence > 0:
        correct = (choice == 1)
    elif coherence < 0:
        correct = (choice == 2)
    else:
        correct = bool(_RNG.random() > 0.5)

    return {
        'choice': choice,
        'reaction_time': decision_time,
        'correct': correct,
        'coherence': coherence,
        'threshold': threshold,
        'r1': r1, 'r2': r2, 'time': T
    }


def demonstrate_threshold_effects():
    """Show the basic story: raise threshold -> slower & (usually) more accurate."""
    net = WongWangNetwork()
    thresholds = [15, 25, 35, 45]     # Hz
    coherence  = 16.0                 # moderate
    n_trials   = 15

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Threshold demo (example trace, RTs, summary)', fontsize=14)

    for col, thr in enumerate(thresholds):
        results = [simulate_trial_with_threshold(net, coherence, thr) for _ in range(n_trials)]
        example = results[0]

        # quick stats
        rts = [r['reaction_time'] for r in results if r['reaction_time'] < 1900]
        acc = np.mean([r['correct'] for r in results])
        mean_rt = np.mean(rts) if len(rts) else np.nan

        # (row 1) example dynamics
        ax = axes[0, col]
        ax.plot(example['time'], example['r1'], 'r', lw=2, alpha=0.8, label='pop 1')
        ax.plot(example['time'], example['r2'], 'b', lw=2, alpha=0.8, label='pop 2')
        ax.axhline(thr, color='k', ls='--', lw=1.8, label=f'threshold={thr}')
        ax.axvline(example['reaction_time'], color='purple', ls=':', alpha=0.7)
        if col == 0: ax.legend()
        ax.set_title(f'{thr} Hz (RT≈{mean_rt:.0f} ms, acc={acc:.2f})')
        ax.set_xlim(0, 1500); ax.set_ylim(0, 60); ax.grid(True, alpha=0.3)

        # (row 2) RT histogram
        ax = axes[1, col]
        if len(rts):
            ax.hist(rts, bins=12, alpha=0.75, edgecolor='black')
            ax.axvline(mean_rt, color='red', lw=2)
        ax.set_xlabel('RT (ms)'); ax.set_ylabel('count'); ax.grid(True, alpha=0.3)

        # (row 3) quick “scoreboard”: accuracy, speed index, confidence
        ax = axes[2, col]
        # rough confidence = final |r1-r2|
        conf = np.mean([abs(np.mean(r['r1'][-50:]) - np.mean(r['r2'][-50:])) for r in results])
        speed = 1000.0 / mean_rt if np.isfinite(mean_rt) else 0.0
        vals  = [acc, speed/2.0, conf/20.0]  # just scaled to sit together
        bars  = ax.bar(['acc', '1/rt', 'conf'], vals, color=['green','steelblue','orange'],
                       edgecolor='black', alpha=0.8)
        for b, v in zip(bars, [acc, speed, conf]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1.2); ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def systematic_tradeoff_analysis():
    """Small grid: thresholds × coherences → speed/accuracy curves + efficiency."""
    net = WongWangNetwork()
    thresholds = [15, 20, 25, 30, 35, 40, 45]
    coherences = [8, 16, 32]
    n_trials   = 12

    results = {}
    t0 = time.time()
    for c in coherences:
        results[c] = {}
        for thr in thresholds:
            trials = [simulate_trial_with_threshold(net, c, thr, max_time=1800) for _ in range(n_trials)]
            rts = [r['reaction_time'] for r in trials if r['reaction_time'] < 1700]
            accs = [r['correct'] for r in trials]
            results[c][thr] = {
                'mean_rt': np.mean(rts) if len(rts) else np.nan,
                'std_rt':  np.std(rts)  if len(rts) else np.nan,
                'mean_acc': np.mean(accs),
                'std_acc':  np.std(accs),
            }
    print(f"grid run in {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Speed–accuracy trade-off (grid summary)', fontsize=14)
    cols = ['tab:blue', 'tab:green', 'tab:red']

    # (1) RT vs threshold
    ax = axes[0,0]
    for i,c in enumerate(coherences):
        ax.errorbar(thresholds, [results[c][t]['mean_rt'] for t in thresholds],
                    yerr=[results[c][t]['std_rt']  for t in thresholds],
                    fmt='o-', color=cols[i], capsize=3, label=f'{c}%')
    ax.set_xlabel('threshold (Hz)'); ax.set_ylabel('RT (ms)'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('threshold → speed')

    # (2) accuracy vs threshold
    ax = axes[0,1]
    for i,c in enumerate(coherences):
        ax.errorbar(thresholds, [results[c][t]['mean_acc'] for t in thresholds],
                    yerr=[results[c][t]['std_acc']  for t in thresholds],
                    fmt='o-', color=cols[i], capsize=3, label=f'{c}%')
    ax.set_xlabel('threshold (Hz)'); ax.set_ylabel('accuracy'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('threshold → accuracy')

    # (3) speed–accuracy curve
    ax = axes[0,2]
    for i,c in enumerate(coherences):
        rts  = [results[c][t]['mean_rt']  for t in thresholds]
        accs = [results[c][t]['mean_acc'] for t in thresholds]
        ax.plot(rts, accs, 'o-', lw=3, ms=6, color=cols[i], alpha=0.85, label=f'{c}%')
    ax.set_xlabel('RT (ms)'); ax.set_ylabel('accuracy'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('speed vs accuracy')

    # (4) simple efficiency metric = accuracy per second
    ax = axes[1,0]
    for i,c in enumerate(coherences):
        eff = []
        for t in thresholds:
            rt = results[c][t]['mean_rt']
            eff.append(results[c][t]['mean_acc'] / (rt/1000.0) if np.isfinite(rt) else 0.0)
        ax.plot(thresholds, eff, 'o-', color=cols[i], label=f'{c}%')
    ax.set_xlabel('threshold (Hz)'); ax.set_ylabel('acc / sec'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('efficiency')

    # (5) “best” thresholds per coherence (max efficiency)
    ax = axes[1,1]
    best_thr = []
    for c in coherences:
        eff = []
        for t in thresholds:
            rt = results[c][t]['mean_rt']
            eff.append(results[c][t]['mean_acc'] / (rt/1000.0) if np.isfinite(rt) else 0.0)
        best_thr.append(thresholds[int(np.argmax(eff))])
    bars = ax.bar([f'{c}%' for c in coherences], best_thr, color=cols, edgecolor='black', alpha=0.8)
    for b, v in zip(bars, best_thr):
        ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v}', ha='center', va='bottom')
    ax.set_ylabel('optimal threshold (Hz)'); ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('argmax(acc/sec)')

    # (6) quick notes panel
    ax = axes[1,2]
    lines = [
        "Takeaways:",
        "• Lower threshold → faster, usually less accurate.",
        "• Higher threshold → slower, usually more accurate.",
        "• Efficiency tends to peak somewhere in the middle.",
        "• Better coherence shifts the sweet spot lower.",
        "• Urgency (next demo) tilts toward speed under time pressure.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va='top', family='monospace')
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def demonstrate_urgency_signals():
    """Keep task fixed; change urgency schedule; watch RT shift."""
    net = WongWangNetwork()
    setups = [
        {'name': 'No Urgency',     'onset': 0,   'slope': 0.0},
        {'name': 'Early Urgency',  'onset': 300, 'slope': 0.01},
        {'name': 'Late Urgency',   'onset': 800, 'slope': 0.02},
        {'name': 'Strong Urgency', 'onset': 400, 'slope': 0.03},
    ]
    coherence = 12.8
    threshold = 30

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Urgency schedules (same task, shifted RT)', fontsize=14)

    for i, cfg in enumerate(setups):
        r = simulate_trial_with_threshold(
            net, coherence, threshold, max_time=1500,
            urgency_onset=cfg['onset'], urgency_slope=cfg['slope']
        )

        # activity traces
        ax = axes[0, i]
        ax.plot(r['time'], r['r1'], 'r', lw=2, alpha=0.8, label='pop 1')
        ax.plot(r['time'], r['r2'], 'b', lw=2, alpha=0.8, label='pop 2')
        ax.axhline(threshold, color='k', ls='--', lw=1.5)
        ax.axvline(r['reaction_time'], color='purple', ls=':', alpha=0.7)
        if cfg['onset'] > 0:
            ax.axvline(cfg['onset'], color='orange', ls='--', alpha=0.6)
        if i == 0: ax.legend()
        ax.set_title(f"{cfg['name']} (RT≈{r['reaction_time']:.0f} ms)")
        ax.set_xlim(0, 1500); ax.set_ylim(0, 50); ax.grid(True, alpha=0.3)

        # the urgency ramp itself (just to see it)
        ax = axes[1, i]
        t = r['time']
        urg = np.zeros_like(t)
        if cfg['slope'] > 0:
            mask = t > cfg['onset']
            urg[mask] = cfg['slope'] * (t[mask] - cfg['onset'])
        ax.plot(t, urg, color='orange', lw=3)
        ax.axvline(r['reaction_time'], color='purple', ls=':', alpha=0.7)
        ax.set_title(f"urgency: onset {cfg['onset']} ms, slope {cfg['slope']}")
        ax.set_xlim(0, 1500); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demonstrate_threshold_effects()
    systematic_tradeoff_analysis()
    demonstrate_urgency_signals()
    print("\nDone with Exercise 4")
    