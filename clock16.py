"""
16-bit stochastic binary clock on a TSU.

Scales the earlier one-hot clock + 8-bit adder up to a real 16-bit counting
clock:

    - TIMING   : the THRML pbit stochastic clock (emergent ticks, one-hot
                 micro-phase) decides *when* the clock advances.
    - DATAPATH : a parametric N-bit energy-based incrementer (the adder
                 gadget with B clamped to the constant 1). Each tick the
                 ground state of the EBM is count + 1.
    - STATE    : a host-carried 16-bit register; the clock's readout is a
                 full 16-bit binary word (range 0..65535).

The 16-bit ripple-carry chain is twice as deep as the 8-bit one, so this is
where FEASIBILITY.md's Wall A (mixing time on a longer logical chain) starts
to bite. The mitigation, also predicted there, is a beta-annealing schedule:
sample hot (wide, fast-mixing) then progressively cold (concentrated on the
ground state), carrying the chain state forward between stages.

Validation (all asserted)
-------------------------
1. Clock one-hot invariant + strictly increasing tick times.
2. Every increment is exact: EBM penalty 0 and value == prev + 1.
3. The register genuinely exceeds 8 bits (the run crosses 255 -> 256, and
   carry-stress probes drive bits up to 2^15).
4. End-to-end: the emitted 16-bit sequence is the contiguous ramp.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from stochastic_clock import draw_pbit_stream, clock_ticks

NBITS = 16


# ---------------------------------------------------------------------------
# Parametric N-bit adder/incrementer EBM (generalised from adder8.py).
# ---------------------------------------------------------------------------
def build_adder_model(nbits: int, beta: float):
    a = [SpinNode() for _ in range(nbits)]
    b = [SpinNode() for _ in range(nbits)]
    s = [SpinNode() for _ in range(nbits)]
    c = [SpinNode() for _ in range(nbits + 1)]

    nodes = a + b + s + c
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    clamped_nodes = a + b + [c[0]]
    free_nodes = s + c[1:]
    free_set = set(free_nodes)
    is_free = {nd: (nd in free_set) for nd in nodes}

    L = np.zeros(n)
    Q: dict[tuple[int, int], float] = {}
    for i in range(nbits):
        terms = [(a[i], 1), (b[i], 1), (c[i], 1), (s[i], -1), (c[i + 1], -2)]
        for nd, coef in terms:
            L[idx[nd]] += coef * coef
        for (n1, c1), (n2, c2) in itertools.combinations(terms, 2):
            i1, i2 = idx[n1], idx[n2]
            lo, hi = (i1, i2) if i1 < i2 else (i2, i1)
            Q[(lo, hi)] = Q.get((lo, hi), 0.0) + 2.0 * c1 * c2

    h = L / 2.0
    J: dict[tuple[int, int], float] = {}
    for (k, l), q in Q.items():
        J[(k, l)] = q / 4.0
        h[k] += q / 4.0
        h[l] += q / 4.0

    edges, weights = [], []
    for (k, l), j in J.items():
        if j == 0.0:
            continue
        if is_free[nodes[k]] or is_free[nodes[l]]:
            edges.append((nodes[k], nodes[l]))
            weights.append(-j)                       # negate: E prop. +H

    model = IsingEBM(
        nodes, edges, jnp.array(-h), jnp.array(weights), jnp.array(beta)
    )

    adj = {nd: set() for nd in free_nodes}
    for (u, v) in edges:
        if is_free[u] and is_free[v]:
            adj[u].add(v)
            adj[v].add(u)
    color: dict = {}
    for nd in free_nodes:
        used = {color[w] for w in adj[nd] if w in color}
        col = 0
        while col in used:
            col += 1
        color[nd] = col
    ncolors = max(color.values()) + 1
    free_blocks = [
        Block([nd for nd in free_nodes if color[nd] == col])
        for col in range(ncolors)
    ]
    clamped_block = Block(clamped_nodes)

    info = dict(a=a, b=b, s=s, c=c, free_nodes=free_nodes,
                free_blocks=free_blocks, clamped_block=clamped_block,
                color=color, ncolors=ncolors, nbits=nbits)
    return model, info


def int_to_bits(x: int, nbits: int) -> list[int]:
    return [(x >> i) & 1 for i in range(nbits)]


def exact_penalty(abits, bbits, sbits, cbits, nbits) -> int:
    H = 0
    for i in range(nbits):
        r = abits[i] + bbits[i] + cbits[i] - sbits[i] - 2 * cbits[i + 1]
        H += r * r
    return H


def _flat_to_color_init(flat_bits: np.ndarray, info):
    """Map a free-vector (free_nodes order) into per-color init arrays."""
    pos = {nd: k for k, nd in enumerate(info["free_nodes"])}
    out = []
    for blk in info["free_blocks"]:
        out.append(jnp.array([bool(flat_bits[pos[nd]]) for nd in blk],
                              dtype=bool))
    return out


def anneal_solve(A: int, B: int, nbits: int, stages, n_chains: int, key):
    """Beta-annealed best-of-chains solve of S = A + B as an EBM ground state.

    `stages` = list of (beta, n_warmup); the last stage also collects samples.
    Chain state is carried forward across rising-beta stages (simulated
    annealing)."""
    abits = int_to_bits(A, nbits)
    bbits = int_to_bits(B, nbits)

    # clamp vector aligned to clamped_block node order (built once)
    _, info0 = build_adder_model(nbits, 1.0)
    clamp_vals = []
    for nd in info0["clamped_block"]:
        if nd in info0["a"]:
            clamp_vals.append(bool(abits[info0["a"].index(nd)]))
        elif nd in info0["b"]:
            clamp_vals.append(bool(bbits[info0["b"].index(nd)]))
        else:
            clamp_vals.append(False)               # carry-in = 0
    clamp_state = [jnp.array(clamp_vals, dtype=bool)]

    best_val, best_H, n_exact, n_total = None, np.inf, 0, 0

    for ch in range(n_chains):
        k = jax.random.fold_in(key, ch)
        # rebuild models per beta (cheap); reuse same node objects per build
        models = [build_adder_model(nbits, beta) for (beta, _) in stages]
        m0, info = models[0]
        k_i, k = jax.random.split(k, 2)
        state = hinton_init(k_i, m0, info["free_blocks"], ())

        for sidx, (beta, nw) in enumerate(stages):
            model, info = models[sidx]
            program = IsingSamplingProgram(
                model, info["free_blocks"], [info["clamped_block"]]
            )
            observe = [Block(info["free_nodes"])]
            last = sidx == len(stages) - 1
            sch = SamplingSchedule(
                n_warmup=nw,
                n_samples=(100 if last else 1),
                steps_per_sample=(4 if last else 1),
            )
            k_s, k = jax.random.split(k, 2)
            out = np.asarray(
                sample_states(k_s, program, sch, state, clamp_state, observe)[0]
            )
            # carry final configuration forward to the next (colder) stage
            state = _flat_to_color_init(out[-1], info)

            if last:
                for row in out:
                    val = {nd: int(bool(row[i]))
                           for i, nd in enumerate(info["free_nodes"])}
                    cbits = [0] + [val[info["c"][i]]
                                   for i in range(1, nbits + 1)]
                    sbits = [val[info["s"][i]] for i in range(nbits)]
                    S = (sum(sbits[i] << i for i in range(nbits))
                         + (cbits[nbits] << nbits))
                    H = exact_penalty(abits, bbits, sbits, cbits, nbits)
                    n_total += 1
                    n_exact += (H == 0)
                    if H < best_H:
                        best_H, best_val = H, S

    return dict(pred=best_val, penalty=best_H,
                ground_hit_rate=n_exact / n_total,
                correct=(best_val == A + B and best_H == 0))


# ---------------------------------------------------------------------------
# Main: stochastic clock drives a 16-bit binary counter.
# ---------------------------------------------------------------------------
P = 0.30
THETA = 6
K = 8
N_MICRO = 6000

STAGES = [(0.5, 600), (1.2, 600), (2.5, 1000), (4.0, 1800)]  # beta anneal
N_CHAINS = 6

START = 250          # seed near the 8-bit boundary
N_TICKS = 12         # crosses 255 -> 256 (proves >8-bit datapath)
STRESS = [4095, 16383, 65534]   # high-bit carry-stress single increments


def fmt16(x: int) -> str:
    return format(x, "016b")


def main():
    print("=" * 78)
    print("16-BIT STOCHASTIC BINARY CLOCK   (pbit timing + EBM incrementer)")
    print("=" * 78)
    _, i0 = build_adder_model(NBITS, 1.0)
    print(f"datapath           : {NBITS}-bit, {len(i0['free_nodes'])} free "
          f"vars, {i0['ncolors']}-color chromatic Gibbs")
    print(f"clock              : pbit P={P}, theta={THETA}, one-hot K={K}")
    print(f"beta anneal        : {[b for b, _ in STAGES]}  "
          f"(chains={N_CHAINS})")
    print()

    key = jax.random.key(0)
    stream = draw_pbit_stream(jax.random.fold_in(key, 1), P, N_MICRO)
    onehot, tick_steps, phases = clock_ticks(stream, THETA, K)

    # Validation 1: clock timing is a valid advancing one-hot counter.
    assert onehot.shape[1] == K
    assert np.all(onehot.sum(axis=1) == 1), "clock one-hot violated"
    assert np.all(np.diff(tick_steps) > 0), "clock not advancing"
    assert len(tick_steps) >= N_TICKS, "not enough clock ticks; raise N_MICRO"

    print(f"{'tick':>4} {'clk@':>6} {'phase':>9} | {'count16':>16} "
          f"{'dec':>6} {'ref':>6} {'gnd%':>6} | res")
    print("-" * 78)

    count = START
    all_ok = True
    seq, ref = [], []
    for t in range(N_TICKS):
        clk = int(tick_steps[t])
        ph = int(phases[t])
        ohs = "".join("1" if j == ph else "0" for j in range(K))

        r = anneal_solve(count, 1, NBITS, STAGES, N_CHAINS,
                         jax.random.fold_in(key, 7000 + t))
        nxt = r["pred"]
        expected = count + 1
        step_ok = (r["penalty"] == 0 and nxt == expected)
        all_ok &= step_ok
        seq.append(nxt)
        ref.append(expected)

        print(f"{t:>4} {clk:>6} {ohs:>9} | {fmt16(nxt):>16} "
              f"{nxt:>6} {expected:>6} {r['ground_hit_rate'] * 100:>5.1f}% | "
              f"{'OK' if step_ok else 'BAD'}")
        count = nxt

    print("-" * 78)
    crossed = any(v > 255 for v in seq)
    print(f"crossed 8-bit boundary (value > 255): "
          f"{'YES' if crossed else 'NO'}  -> genuine 16-bit datapath")
    assert crossed, "did not exceed 255; not exercising >8-bit"

    # Validation 3: high-bit carry-stress single increments.
    print()
    print("carry-stress single increments (drive the upper bits):")
    print(f"{'A':>7} +1 -> {'pred':>7} {'binary16':>16} {'gnd%':>6} | res")
    print("-" * 78)
    stress_ok = True
    for v in STRESS:
        r = anneal_solve(v, 1, NBITS, STAGES, N_CHAINS,
                         jax.random.fold_in(key, 99000 + v))
        ok = r["correct"] and r["pred"] == v + 1
        stress_ok &= ok
        print(f"{v:>7} +1 -> {r['pred']:>7} {fmt16(r['pred']):>16} "
              f"{r['ground_hit_rate'] * 100:>5.1f}% | {'OK' if ok else 'BAD'}")

    print("-" * 78)
    seq_ok = seq == ref
    print(f"contiguous ramp {START + 1}..{START + N_TICKS} : "
          f"{'YES' if seq_ok else 'NO'}")
    print(f"per-tick arithmetic exact        : "
          f"{'YES' if all_ok else 'NO'}")
    print(f"carry-stress (bits up to 2^15)   : "
          f"{'YES' if stress_ok else 'NO'}")
    print()
    ok = all_ok and seq_ok and stress_ok and crossed
    print("RESULT:", "ALL VALIDATIONS PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
