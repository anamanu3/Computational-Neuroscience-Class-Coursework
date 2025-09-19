#Q7:
from pathlib import Path
import pickle, numpy as np
import matplotlib.pyplot as plt

# Folder where your file lives
base = Path("/Users/anamariamanu/Intro to Comp Neuro/quiz4")

# Exact filename (with spaces and parentheses handled safely by Path)
fp = base / "_e96f8f1cf1f8256c8595dcb9668fee4f_tuning_3.4 (1).pickle"

print("Using:", fp)

# Load the pickle
with open(fp, "rb") as f:
    data = pickle.load(f)

# Extract stimulus
stim = np.array(data["stim"]).flatten()
order = np.argsort(stim)
stim = stim[order]

# Plot tuning curves
for k in ["neuron1", "neuron2", "neuron3", "neuron4"]:
    m = np.array(data[k]).mean(axis=0)[order]   # mean firing rate
    plt.figure()
    plt.plot(stim, m, marker="o")
    plt.xlabel("Stimulus direction (deg)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title(f"{k} tuning curve")
    plt.show()





#Q8:
# poisson_check.py
from pathlib import Path
import pickle
import numpy as np

# >>> edit this folder to your local path
BASE = Path("/Users/anamariamanu/Intro to Comp Neuro/quiz4")

# find the tuning file (handles the long randomized prefix)
cand = sorted(BASE.glob("*tuning*3.4*.pickle"))
if not cand:
    raise FileNotFoundError("Couldn't find a *tuning*3.4*.pickle in BASE.")
tuning_path = cand[0]
print("Using:", tuning_path)

with open(tuning_path, "rb") as f:
    tuning = pickle.load(f)

T = 10.0  # seconds per trial (given in the quiz)
expect_ratio = 1.0 / T  # ~0.1 for Poisson

ratios = {}
for k in ["neuron1", "neuron2", "neuron3", "neuron4"]:
    # M: shape (n_trials, n_stimuli), entries are firing rates (Hz)
    M = np.array(tuning[k], dtype=float)
    mean_per_stim = M.mean(axis=0)
    var_per_stim  = M.var(axis=0, ddof=1)
    # Var(rate)/Mean(rate) for each stimulus (ignore zeros to avoid div-by-zero warnings)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = var_per_stim / mean_per_stim
    ratios[k] = float(np.nanmean(ratio))

print("\nExpected Var/Mean for Poisson rates ≈", expect_ratio)
for k, r in ratios.items():
    print(f"{k:7s}  Var/Mean ≈ {r:.3f}")

# Identify the oddball (largest deviation from 0.1)
oddball = max(ratios, key=lambda k: abs(ratios[k] - expect_ratio))
print("\nMost non-Poisson (largest deviation):", oddball)





#Q9:
# population_decode.py
from pathlib import Path
import pickle
import numpy as np

BASE = Path("/Users/anamariamanu/Intro to Comp Neuro/quiz4")

def find_one(base: Path, prefer: list[str], fallback_globs: list[str]):
    for nm in prefer:
        p = base / nm
        if p.exists():
            return p

    cands = []
    for pat in fallback_globs:
        cands += list(base.glob(pat))
   
    cands = sorted(set(cands), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        have = [p.name for p in base.glob("*")]
        raise FileNotFoundError(
            f"No match in {base}\nTried patterns: {fallback_globs}\nFiles present:\n" +
            "\n".join("  " + h for h in have)
        )
    return cands[0]

tuning_exact   = [
    "_e96f8f1cf1f8256c8595dcb9668fee4f_tuning_3.4.pickle",
    "_e96f8f1cf1f8256c8595dcb9668fee4f_tuning_3.4 (1).pickle",
]
pop_exact      = [
    "_e96f8f1cf1f8256c8595dcb9668fee4f_pop_coding_3.4.pickle",
    "_e96f8f1cf1f8256c8595dcb9668fee4f_pop_coding_3.4 (1).pickle",
]

tuning_globs   = ["*tuning*3.4*.pickle", "*tuning*.pickle"]
pop_globs      = ["*pop*cod*3.4*.pickle", "*pop*cod*.pickle", "*coding*3.4*.pickle"]

tuning_path = find_one(BASE, tuning_exact, tuning_globs)
pop_path    = find_one(BASE, pop_exact,    pop_globs)

print("Using tuning:     ", tuning_path.name)
print("Using pop_coding: ", pop_path.name)


with open(tuning_path, "rb") as f:
    tuning = pickle.load(f)
with open(pop_path, "rb") as f:
    pop = pickle.load(f)

# r_max for each neuron from tuning curves (mean over trials, max over stimuli
rmax = {}
for k in ["neuron1", "neuron2", "neuron3", "neuron4"]:
    M = np.array(tuning[k], dtype=float)     # shape: (trials, stimuli)
    rmax[k] = float(M.mean(axis=0).max())

#Mean response to the mystery stimulus (10 trials)
rbar = {
    "neuron1": float(np.mean(pop["r1"])),
    "neuron2": float(np.mean(pop["r2"])),
    "neuron3": float(np.mean(pop["r3"])),
    "neuron4": float(np.mean(pop["r4"])),
}

#basis (preferred direction) vectors c1..c4
c = {
    "neuron1": np.array(pop["c1"], dtype=float),
    "neuron2": np.array(pop["c2"], dtype=float),
    "neuron3": np.array(pop["c3"], dtype=float),
    "neuron4": np.array(pop["c4"], dtype=float),
}

# Population vector: sum_i (r_i / r_max_i) * c_i
v = sum((rbar[k] / rmax[k]) * c[k] for k in rmax.keys())

#angle
ang_std  = np.degrees(np.arctan2(v[1], v[0])) % 360
ang_quiz = (90.0 - ang_std) % 360.0

print(f"\nDecoded direction (quiz convention): {ang_quiz:.6f}°")
print("Rounded to nearest degree:", int(np.round(ang_quiz)))


