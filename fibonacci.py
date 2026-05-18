"""
Fibonacci on a TSU: stochastic clock  +  energy-based adder.

This is the two earlier primitives composed into a real program, using the
host-sequenced time-slicing pattern recommended in FEASIBILITY.md:

    - TIMING  (emergent "program counter"):  stochastic_clock.py
        A THRML pbit accumulates random events; every `theta` firings it
        emits a one-hot phase tick. The tick is the *only* thing that says
        "advance the computation" -- time is emergent, not an external clock.

    - COMPUTE (the arithmetic kernel):       adder8.py
        Each tick triggers one relaxation of the 8-bit adder energy-based
        model whose ground state is S = A + B.

    - STATE   (host-carried registers):
        Two 8-bit registers (a, b) hold F(n-2), F(n-1). The classical host
        shifts them between relaxations:  (a, b) <- (b, a+b).

So: clock tick -> solve EBM for a+b -> shift registers -> repeat. Fibonacci
is intrinsically sequential (F(n) needs F(n-1), F(n-2)); a single static
energy landscape cannot express the unbounded recurrence, but slicing it per
step -- each slice a wide-basin 8-bit add -- stays firmly in the tractable
class. 8-bit operands cap the run at the largest term <= 255 (F = 233).

Validation (all asserted)
-------------------------
1. Clock: one-hot invariant + strictly increasing tick times (timing works).
2. Adder: every step reaches penalty 0 and S == a+b (exact arithmetic from
   the EBM ground state).
3. End-to-end: the emitted sequence equals the reference Fibonacci numbers.
"""

import jax
import numpy as np

from thrml import SamplingSchedule

# Reuse the two validated primitives verbatim.
from stochastic_clock import draw_pbit_stream, clock_ticks
from adder8 import solve as adder_solve

# ---- knobs ----------------------------------------------------------------
P = 0.30          # pbit firing probability (clock noise source)
THETA = 6         # firings accumulated per clock tick
K = 8             # one-hot phase register width
N_MICRO = 6000    # micro-steps of pbit stream to draw

BETA = 2.5
SCHEDULE = SamplingSchedule(n_warmup=3000, n_samples=120, steps_per_sample=4)
N_CHAINS = 8
MAX8 = 255        # 8-bit operand ceiling


def reference_fib(limit: int) -> list[int]:
    """Reference Fibonacci terms produced by the (a,b)<-(b,a+b) recurrence,
    starting (a,b)=(0,1), while the new term fits in 8 bits."""
    out, a, b = [], 0, 1
    while a + b <= limit:
        out.append(a + b)
        a, b = b, a + b
    return out


def main():
    print("=" * 76)
    print("FIBONACCI ON A TSU   (stochastic clock drives an energy-based adder)")
    print("=" * 76)
    print(f"clock : pbit P(fire)={P}, theta={THETA}, one-hot K={K}")
    print(f"adder : beta={BETA}, warmup={SCHEDULE.n_warmup}, "
          f"samples={SCHEDULE.n_samples}, chains={N_CHAINS}")
    print()

    key = jax.random.key(0)

    # ---- emergent timing: draw one pbit stream, extract one-hot ticks ----
    stream = draw_pbit_stream(jax.random.fold_in(key, 1), P, N_MICRO)
    onehot, tick_steps, phases = clock_ticks(stream, THETA, K)

    # Validation 1: the clock is a valid one-hot, strictly advancing counter.
    assert onehot.shape[1] == K
    assert np.all(onehot.sum(axis=1) == 1), "clock one-hot violated"
    assert np.all(np.diff(tick_steps) > 0), "clock not advancing"

    ref = reference_fib(MAX8)
    n_steps = len(ref)
    assert len(tick_steps) >= n_steps, (
        f"clock produced only {len(tick_steps)} ticks, need {n_steps}; "
        f"increase N_MICRO")

    print(f"{'step':>4} {'clk@':>6} {'phase(one-hot)':>16} | "
          f"{'A':>4} {'B':>4} -> {'S(TSU)':>7} {'ref':>5} {'gnd%':>6} | res")
    print("-" * 76)

    # ---- run: each clock tick triggers one adder relaxation ----
    a, b = 0, 1
    produced = []
    all_ok = True
    for step in range(n_steps):
        clk = int(tick_steps[step])
        ph = int(phases[step])
        onehot_str = "".join("1" if j == ph else "0" for j in range(K))

        r = adder_solve(a, b, BETA, SCHEDULE, N_CHAINS,
                        jax.random.fold_in(key, 1000 + step))
        S = r["pred"]

        # Validation 2: exact arithmetic from the EBM ground state.
        step_ok = (r["penalty"] == 0 and S == a + b)
        all_ok &= step_ok
        produced.append(S)

        print(f"{step:>4} {clk:>6} {onehot_str:>16} | "
              f"{a:>4} {b:>4} -> {S:>7} {ref[step]:>5} "
              f"{r['ground_hit_rate'] * 100:>5.1f}% | "
              f"{'OK' if step_ok else 'BAD'}")

        a, b = b, S          # host shifts registers between relaxations

    print("-" * 76)

    # Validation 3: end-to-end sequence correctness.
    seq_ok = produced == ref
    print(f"sequence produced : {produced}")
    print(f"reference Fibonacci: {ref}")
    print(f"next term {a}+{b}={a + b} > {MAX8}  -> halt (8-bit operand limit)")
    print()
    print(f"per-step arithmetic : {'all exact' if all_ok else 'ERROR'}")
    print(f"full sequence match : {'YES' if seq_ok else 'NO'}")
    print()
    print("RESULT:", "ALL VALIDATIONS PASSED"
          if (all_ok and seq_ok) else "FAILED")
    raise SystemExit(0 if (all_ok and seq_ok) else 1)


if __name__ == "__main__":
    main()
