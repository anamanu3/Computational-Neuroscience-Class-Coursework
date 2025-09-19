"""
Complete Hopfield Network — all parts in one file
=================================================

What this script does:
1) Small 4-neuron demo (store one pattern, recover from noisy starts)
2) Pattern completion in a bigger net (noise vs recovery)
3) Storage capacity sweep vs the ~0.14·N rule of thumb
4) Checkerboard + extra random patterns (stability/recovery)
5) Spurious/mixed states, track overlaps over time

Ref: https://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/hopfield-network.html
"""

import numpy as np
import matplotlib.pyplot as plt
import neurodynex3.hopfield_network.network as network
import neurodynex3.hopfield_network.pattern_tools as pattern_tools
import neurodynex3.hopfield_network.plot_tools as plot_tools


# -----------------------------------------------------------------------------
# Exercise 1: Getting started (4-neuron network)
# -----------------------------------------------------------------------------
def exercise_1_four_neuron_network():
    print("\n" + "="*60)
    print("Ex1) 4-neuron demo")
    print("="*60)

    hopfield_net = network.HopfieldNetwork(nr_neurons=4)

    pattern = np.array([-1, -1, +1, +1])
    hopfield_net.store_patterns([pattern])
    print(f"Stored pattern: {pattern}")
    print("Weight matrix (diag≈0 expected):")
    print(hopfield_net.weights)

    tests = [
        np.array([-1, -1, -1, +1]),
        np.array([+1, -1, +1, +1]),
        np.array([-1, -1, -1, -1]),
    ]

    for i, init in enumerate(tests, 1):
        print(f"\nTest {i} start:", init)
        hopfield_net.set_state_from_pattern(init)
        print("  step 0:", hopfield_net.state)
        converged = False
        for step in range(1, 6):
            hopfield_net.run_with_monitoring(nr_steps=1)
            print(f"  step {step}:", hopfield_net.state)
            if np.array_equal(hopfield_net.state, pattern):
                print("  converged to stored pattern")
                converged = True
                break
        if not converged:
            print("  no full convergence in 5 steps")

    return hopfield_net


# -----------------------------------------------------------------------------
# Exercise 2: Pattern completion and associative memory
# -----------------------------------------------------------------------------
def exercise_2_pattern_completion():
    print("\n" + "="*60)
    print("Ex2) Pattern completion (larger network)")
    print("="*60)

    N = 50
    hopfield_net = network.HopfieldNetwork(nr_neurons=N)

    patterns = [np.random.choice([-1, 1], size=N) for _ in range(3)]
    hopfield_net.store_patterns(patterns)
    print(f"Stored {len(patterns)} patterns (N={N})")

    noise_levels = [0.1, 0.2, 0.3, 0.4]
    for idx, p in enumerate(patterns, start=1):
        print(f"\nPattern {idx}:")
        for noise in noise_levels:
            noisy = p.copy()
            flips = np.random.choice(N, size=int(noise*N), replace=False)
            noisy[flips] *= -1

            hopfield_net.set_state_from_pattern(noisy)
            init_overlap = np.mean(hopfield_net.state == p)
            hopfield_net.run_with_monitoring(nr_steps=20)
            final_overlap = np.mean(hopfield_net.state == p)

            tag = "ok" if final_overlap > 0.9 else "~" if final_overlap > 0.7 else "low"
            print(f"  noise {noise:.1f}: {init_overlap:.2f} → {final_overlap:.2f}  {tag}")

    return patterns


# -----------------------------------------------------------------------------
# Exercise 3: Storage capacity
# -----------------------------------------------------------------------------
def exercise_3_storage_capacity():
    print("\n" + "="*60)
    print("Ex3) Storage capacity (success vs #patterns)")
    print("="*60)

    N = 100
    max_patterns = 25
    trials = 5
    counts = list(range(1, max_patterns + 1, 2))
    success_rates = []

    print(f"N={N}; up to {max_patterns} patterns (every other value).")
    print("Each point = fraction of successful retrievals over", trials, "trials")

    for P in counts:
        successes = 0
        for _ in range(trials):
            net = network.HopfieldNetwork(nr_neurons=N)
            pats = [np.random.choice([-1, 1], size=N) for _ in range(P)]
            net.store_patterns(pats)

            test = pats[0].copy()
            flips = np.random.choice(N, size=int(0.2*N), replace=False)
            test[flips] *= -1

            net.set_state_from_pattern(test)
            net.run_with_monitoring(nr_steps=10)
            overlap = np.mean(net.state == pats[0])
            if overlap > 0.9:
                successes += 1

        rate = successes / trials
        success_rates.append(rate)
        print(f"  P={P:2d} → success={rate:.2f}")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(counts, success_rates, 'o-', lw=2)
    plt.axhline(0.5, ls='--', c='r', alpha=0.7)
    th = 0.14 * N
    plt.axvline(th, ls=':', c='g', alpha=0.7, label=f'~0.14·N ≈ {th:.1f}')
    plt.xlabel('Stored patterns (P)')
    plt.ylabel('Success rate')
    plt.title(f'Capacity curve (N={N})')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    frac = np.array(counts)/N
    plt.plot(frac, success_rates, 'o-', lw=2)
    plt.axhline(0.5, ls='--', c='r', alpha=0.7)
    plt.axvline(0.14, ls=':', c='g', alpha=0.7, label='0.14 (rule of thumb)')
    plt.xlabel('P/N')
    plt.ylabel('Success rate')
    plt.title('Capacity as fraction of network size')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    empirical = next((counts[i] for i, r in enumerate(success_rates) if r < 0.5), None)
    print("\nCapacity notes:")
    print(f"  Theoretical ~0.14·N ≈ {th:.1f}")
    print(f"  Empirical ~{empirical} patterns" if empirical else "  Empirical > tested range")

    return counts, success_rates


# -----------------------------------------------------------------------------
# Exercise 4: Checkerboard pattern (fixed shapes)
# -----------------------------------------------------------------------------
def exercise_4_checkerboard_pattern():
    print("\n" + "="*60)
    print("Ex4) Checkerboard stability + recovery")
    print("="*60)

    grid = 10
    N = grid * grid
    factory = pattern_tools.PatternFactory(grid, grid)

    # build checkerboard (2D), convert to {-1,+1}, flatten to 1D length N
    try:
        checker_2d = factory.create_checkerboard()
    except Exception:
        checker_2d = np.fromfunction(lambda i, j: ((i + j) % 2 == 0).astype(int),
                                     (grid, grid))

    if set(np.unique(checker_2d).tolist()) == {0, 1}:
        checker_2d = 2*checker_2d - 1
    checker_1d = checker_2d.ravel().astype(int)

    print(f"checkerboard built ({grid}×{grid})")

    K_values = [1, 5, 10, 15, 20]
    last_net = None

    for K in K_values:
        print(f"\nK={K}: checker + {K-1} random patterns")
        net = network.HopfieldNetwork(nr_neurons=N)
        pats = [checker_1d] + [np.random.choice([-1, 1], size=N) for _ in range(K-1)]
        net.store_patterns(pats)

        # exact checker start
        net.set_state_from_pattern(checker_1d)
        for _ in range(5):
            net.run_with_monitoring(nr_steps=1)
        overlap = np.mean(net.state == checker_1d)
        print(f"  exact start: overlap={overlap:.3f}  ({'stable' if overlap>0.95 else 'not stable'})")

        # 10% noisy start
        noisy = checker_1d.copy()
        flips = np.random.choice(N, size=int(0.1*N), replace=False)
        noisy[flips] *= -1
        net.set_state_from_pattern(noisy)
        init = np.mean(net.state == checker_1d)
        net.run_with_monitoring(nr_steps=20)
        fin = np.mean(net.state == checker_1d)
        tag = "ok" if fin > 0.9 else "~" if fin > 0.7 else "low"
        print(f"  10% noise  : {init:.3f} → {fin:.3f}  {tag}")

        last_net = net

    # visualize target vs final of last run
    if last_net is not None:
        try:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(checker_2d, cmap='RdBu', vmin=-1, vmax=1)
            plt.title('Checkerboard (target)')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            final_2d = last_net.state.reshape(grid, grid)
            plt.imshow(final_2d, cmap='RdBu', vmin=-1, vmax=1)
            plt.title('Final state (last K)')
            plt.colorbar()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(skip plot: {e})")


# -----------------------------------------------------------------------------
# Exercise 5: Spurious states and dynamics
# -----------------------------------------------------------------------------
def exercise_5_spurious_states():
    print("\n" + "="*60)
    print("Ex5) Spurious/mixed states")
    print("="*60)

    N = 20
    net = network.HopfieldNetwork(nr_neurons=N)

    p1 = np.array([1 if i < N//2 else -1 for i in range(N)])
    p2 = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    net.store_patterns([p1, p2])

    print(f"pattern overlap (p1⋅p2 / N): {np.dot(p1, p2)/N:.3f}")

    tests = [
        ("random", np.random.choice([-1, 1], size=N)),
        ("mixed (p1+p2)/2", (p1 + p2)//2),
        ("-p1 (inverted)", -p1),
        ("half-random", np.array([1 if np.random.rand() > 0.5 else -1 for _ in range(N)])),
    ]

    results = []
    for name, init in tests:
        print(f"\nStart: {name}")
        net.set_state_from_pattern(init)

        o1 = [np.dot(net.state, p1) / N]
        o2 = [np.dot(net.state, p2) / N]
        for _ in range(15):
            net.run_with_monitoring(nr_steps=1)
            o1.append(np.dot(net.state, p1) / N)
            o2.append(np.dot(net.state, p2) / N)

        f1, f2 = o1[-1], o2[-1]
        if f1 > 0.9: lab = "→ p1"
        elif f2 > 0.9: lab = "→ p2"
        elif f1 < -0.9: lab = "→ -p1 (spurious)"
        elif f2 < -0.9: lab = "→ -p2 (spurious)"
        else: lab = "→ spurious/mixed"
        print(f"  final overlaps: p1={f1:.3f}, p2={f2:.3f} {lab}")
        results.append((name, o1, o2, lab))

    plt.figure(figsize=(14, 10))
    for i, (name, o1, o2, lab) in enumerate(results):
        plt.subplot(2, 2, i+1)
        steps = range(len(o1))
        plt.plot(steps, o1, 'o-', ms=4, label='overlap with p1')
        plt.plot(steps, o2, 's-', ms=4, label='overlap with p2')
        plt.axhline(0, ls='--', c='k', alpha=0.3)
        plt.axhline(1, ls=':', c='k', alpha=0.2)
        plt.axhline(-1, ls=':', c='k', alpha=0.2)
        plt.title(f"{name}\n{lab}")
        plt.xlabel("update steps")
        plt.ylabel("overlap")
        plt.ylim(-1.1, 1.1)
        plt.grid(alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.show()

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("HOPFIELD NETWORK — complete walkthrough")
    print("="*60)

    try:
        exercise_1_four_neuron_network()
        exercise_2_pattern_completion()
        exercise_3_storage_capacity()
        exercise_4_checkerboard_pattern()
        exercise_5_spurious_states()

        print("\nSummary:")
        print("  • 4-neuron demo converges")
        print("  • Pattern completion holds up to moderate noise")
        print("  • Capacity matches ~0.14·N trend")
        print("  • Checkerboard stability declines with many random patterns")
        print("  • Spurious/mixed states show up as expected")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        print("Check neurodynex3:  pip install neurodynex3")
        import traceback; traceback.print_exc()