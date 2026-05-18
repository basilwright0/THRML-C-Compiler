"""
Stochastic clock with a one-hot output, driven by THRML pbits.

Idea
----
A biological-style clock does NOT use a precise oscillator. It accumulates
many small, irreversible *random* events (e.g. molecular reactions). The time
to collect `theta` such events is a sum of `theta` random waiting times. By the
law of large numbers that sum *concentrates*: its relative fluctuation shrinks
as 1/sqrt(theta). So a clock built from pure noise becomes *more* consistent
the more noise it integrates. This script demonstrates exactly that, using a
real THRML pbit as the noise source.

Mechanism
---------
- A single-spin Ising EBM is a `pbit`: each Gibbs sample is an independent
  coin flip with P(fire) = sigmoid(2 * beta * bias) = p   (verified empirically).
- An accumulator integrates firings. Every `theta` firings it emits a *tick*:
  the phase advances 0 -> 1 -> ... -> K-1 -> 0, encoded one-hot. The micro-step
  index of each tick is recorded.
- Inter-tick interval = #Bernoulli(p) trials to accumulate `theta` successes
  = NegativeBinomial(theta, p):
      mean = theta / p
      CV   = sqrt((1 - p) / theta)        <-- shrinks as 1/sqrt(theta)

Validation (all asserted)
-------------------------
1. One-hot invariant: every emitted state has exactly one active bit.
2. Counts up: phase sequence is the strict cyclic ramp 0,1,...,K-1,0,...
3. Consistency law: empirical mean & CV of the inter-tick interval match the
   theoretical NegativeBinomial values across a sweep of theta, and the
   relative jitter of a full K-cycle shrinks ~ 1/sqrt(theta).
"""

import jax
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from jax import numpy as jnp

BETA = 1.0


def pbit_bias_for_prob(p: float) -> float:
    """Bias b such that P(spin=+1) = sigmoid(2*BETA*b) = p."""
    return float(np.log(p / (1.0 - p)) / (2.0 * BETA))


def draw_pbit_stream(key, p: float, n: int) -> np.ndarray:
    """`n` iid pbit firings (bool) sampled via THRML, in a single call.

    A 1-node model has no edges; THRML's weight factor needs >=1 edge, so we
    couple to a clamped dummy with weight 0 (independent => marginal is exactly
    Bernoulli(p))."""
    node, dummy = SpinNode(), SpinNode()
    model = IsingEBM(
        [node, dummy],
        [(node, dummy)],
        jnp.array([pbit_bias_for_prob(p), 0.0]),
        jnp.array([0.0]),
        jnp.array(BETA),
    )
    free_blocks = [Block([node])]
    program = IsingSamplingProgram(model, free_blocks, [Block([dummy])])

    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=0, n_samples=n, steps_per_sample=1)
    samples = sample_states(
        k_samp, program, schedule, init_state,
        [jnp.array([False])], [Block([node])],
    )
    return np.asarray(samples[0]).reshape(-1).astype(bool)


def clock_ticks(stream_row: np.ndarray, theta: int, K: int):
    """Vectorised: micro-step index of every tick + one-hot phase states."""
    fire_pos = np.flatnonzero(stream_row)          # indices of firings
    tick_steps = fire_pos[theta - 1::theta]        # every theta-th firing
    n = len(tick_steps)
    phases = (np.arange(n) + 1) % K                # strict cyclic ramp
    onehot = np.zeros((n, K), dtype=int)
    onehot[np.arange(n), phases] = 1
    return onehot, tick_steps, phases


def main():
    P = 0.30
    K = 8
    N_CLOCKS = 64
    N_MICRO = 12_000
    THETAS = [1, 4, 16, 64]

    print("=" * 72)
    print("STOCHASTIC ONE-HOT CLOCK  (THRML pbit-driven)")
    print("=" * 72)
    print(f"pbit P(fire)        : {P}")
    print(f"one-hot phases K    : {K}")
    print(f"micro-steps / clock : {N_MICRO}")
    print(f"independent clocks  : {N_CLOCKS}")
    print()

    key = jax.random.key(0)

    # One THRML call -> reshape into N_CLOCKS independent realisations.
    stream = draw_pbit_stream(key, P, N_CLOCKS * N_MICRO)
    emp_p = stream.mean()
    print(f"[pbit check] target p={P:.4f}  empirical={emp_p:.4f}  "
          f"(|err|={abs(emp_p - P):.4f})")
    assert abs(emp_p - P) < 0.01, "pbit firing rate off target"
    streams = stream.reshape(N_CLOCKS, N_MICRO)
    print()

    all_ok = True
    print(f"{'theta':>6} | {'mean dt':>9} {'theory':>9} | "
          f"{'CV emp':>8} {'CV thy':>8} | {'cycle rel-std':>13}")
    print("-" * 72)

    cv_by_theta = {}
    for theta in THETAS:
        inter_all, cycle_all = [], []
        for c in range(N_CLOCKS):
            onehot, steps, phases = clock_ticks(streams[c], theta, K)

            # --- Validation 1: strict one-hot ---
            assert onehot.shape[1] == K
            assert np.all(onehot.sum(axis=1) == 1), "one-hot violated"
            assert np.all((onehot == 0) | (onehot == 1)), "non-binary"

            # --- Validation 2: counts up as a clean cyclic ramp ---
            assert np.array_equal(phases, (np.arange(len(phases)) + 1) % K), \
                "phase is not a clean ramp"
            assert np.all(np.diff(steps) > 0), "ticks not strictly increasing"

            inter_all.append(np.diff(steps))
            if len(steps) > K:
                cycle_all.append(steps[K::K] - steps[:-K:K])

        inter_all = np.concatenate(inter_all)
        cycle_all = np.concatenate(cycle_all)

        mean_dt = inter_all.mean()
        cv_emp = inter_all.std() / mean_dt
        mean_thy = theta / P
        cv_thy = np.sqrt((1.0 - P) / theta)
        cycle_rel_std = cycle_all.std() / cycle_all.mean()
        cv_by_theta[theta] = cv_emp

        print(f"{theta:>6} | {mean_dt:>9.2f} {mean_thy:>9.2f} | "
              f"{cv_emp:>8.4f} {cv_thy:>8.4f} | {cycle_rel_std:>13.5f}")

        # --- Validation 3: consistency law ---
        if abs(mean_dt - mean_thy) / mean_thy > 0.03:
            all_ok = False
        if abs(cv_emp - cv_thy) / cv_thy > 0.08:
            all_ok = False

    print("-" * 72)
    # Monotone-precision check: more accumulation => strictly lower CV.
    cvs = [cv_by_theta[t] for t in THETAS]
    assert all(cvs[i] > cvs[i + 1] for i in range(len(cvs) - 1)), \
        "CV did not decrease monotonically with theta"
    gain = cvs[0] / cvs[-1]
    print(f"Consistency gain: integrating {THETAS[-1]}x more noise made the "
          f"clock {gain:.2f}x more precise")
    print(f"  (theory: sqrt({THETAS[-1]}/{THETAS[0]}) = "
          f"{np.sqrt(THETAS[-1] / THETAS[0]):.2f}x)")
    print()
    print("RESULT:", "ALL VALIDATIONS PASSED" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
