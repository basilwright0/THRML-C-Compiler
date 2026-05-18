"""
8-bit binary adder as a THRML energy-based model (constraint compiler style).

Idea
----
Addition is not "executed". It is encoded as an energy function whose global
minimum (energy 0) is exactly the configuration where A + B = S. The TSU then
*samples* low-energy states; the correct sum is the ground state.

Encoding
--------
Ripple-carry, one penalty per full-adder stage i = 0..7:

    a_i + b_i + c_i  ==  s_i + 2 * c_{i+1}

enforced by the quadratic penalty

    H_i = (a_i + b_i + c_i - s_i - 2 c_{i+1})^2          (binary vars in {0,1})

H = sum_i H_i is a QUBO (only linear + pairwise terms because x^2 = x for
binary x), so it maps to a 2-local Ising model with NO auxiliary variables.
Ground state H = 0  <=>  the bits encode a correct 8+8 -> 9-bit addition.

- a_i, b_i : clamped inputs (8 + 8 bits)
- c_0      : clamped to 0 (carry-in)
- s_0..s_7 : free  (sum bits, weight 2^i)
- c_1..c_8 : free  (carries; c_8 is the 9th/MSB result bit, weight 256)

The free-variable interaction graph contains triangles (s_i, c_i, c_{i+1}),
so it is NOT 2-colorable. We greedily color it and use one block-Gibbs color
phase per color (chromatic Gibbs) -- valid because each color is an
independent set.

Sign convention (verified against thrml source)
-----------------------------------------------
THRML samples P ~ exp(-E) with  E = -beta * (sum b_i s_i + sum J_ij s_i s_j),
spins s in {-1,+1}. We want to MINIMISE H, so we set the Ising parameters to
the *negatives* of H's Ising coefficients, giving E proportional to +H.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

NBITS = 8


# ----------------------------------------------------------------------------
# Build the QUBO -> Ising model for an 8-bit adder.
# ----------------------------------------------------------------------------
def build_adder_model(beta: float):
    a = [SpinNode() for _ in range(NBITS)]
    b = [SpinNode() for _ in range(NBITS)]
    s = [SpinNode() for _ in range(NBITS)]
    c = [SpinNode() for _ in range(NBITS + 1)]  # c[0] carry-in .. c[8] MSB

    nodes = a + b + s + c
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    clamped_nodes = a + b + [c[0]]
    free_nodes = s + c[1:]
    is_free = {nd: (nd in set(free_nodes)) for nd in nodes}

    # --- accumulate binary QUBO: H = sum L_k x_k + sum_{k<l} Q_kl x_k x_l ---
    L = np.zeros(n)                       # linear (binary)
    Q = {}                               # pairwise (binary), key = (i<j)

    for i in range(NBITS):
        # term list: (node, coefficient) for stage i's residual expression
        terms = [(a[i], 1), (b[i], 1), (c[i], 1), (s[i], -1), (c[i + 1], -2)]
        for nd, coef in terms:
            L[idx[nd]] += coef * coef            # coef^2 * x  (x^2 = x)
        for (n1, c1), (n2, c2) in itertools.combinations(terms, 2):
            i1, i2 = idx[n1], idx[n2]
            lo, hi = (i1, i2) if i1 < i2 else (i2, i1)
            Q[(lo, hi)] = Q.get((lo, hi), 0.0) + 2.0 * c1 * c2

    # --- binary -> spin:  x = (1+sigma)/2 ---
    #   h_k = L_k/2 + sum_{l~k} Q_kl/4 ;  J_kl = Q_kl/4 ; plus a constant
    h = L / 2.0
    J = {}
    for (k, l), q in Q.items():
        J[(k, l)] = q / 4.0
        h[k] += q / 4.0
        h[l] += q / 4.0

    # Keep only edges that influence a free variable's conditional (>=1 free
    # endpoint); clamped-clamped edges are constant and irrelevant to sampling.
    edges, weights = [], []
    for (k, l), j in J.items():
        if j == 0.0:
            continue
        if is_free[nodes[k]] or is_free[nodes[l]]:
            edges.append((nodes[k], nodes[l]))
            weights.append(-j)                   # negate: E prop. +H

    biases = -h                                  # negate: E prop. +H

    model = IsingEBM(
        nodes,
        edges,
        jnp.array(biases),
        jnp.array(weights),
        jnp.array(beta),
    )

    # --- greedily color the FREE-variable interaction graph ---
    adj = {nd: set() for nd in free_nodes}
    for (u, v) in edges:
        if is_free[u] and is_free[v]:
            adj[u].add(v)
            adj[v].add(u)
    color = {}
    for nd in free_nodes:                        # deterministic order
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

    info = dict(
        a=a, b=b, s=s, c=c, nodes=nodes, idx=idx,
        free_nodes=free_nodes, clamped_nodes=clamped_nodes,
        free_blocks=free_blocks, clamped_block=clamped_block,
        ncolors=ncolors,
    )
    return model, info


def int_to_bits(x: int, nbits: int) -> list[int]:
    return [(x >> i) & 1 for i in range(nbits)]


def exact_penalty(abits, bbits, sbits, cbits) -> int:
    """H computed directly from integer bits; 0 iff addition is correct."""
    H = 0
    for i in range(NBITS):
        r = abits[i] + bbits[i] + cbits[i] - sbits[i] - 2 * cbits[i + 1]
        H += r * r
    return H


def decode(sample_bits: np.ndarray, info, A: int, B: int):
    """Map a sampled free configuration back to integers + penalty."""
    val = {nd: int(bool(sample_bits[k]))
           for k, nd in enumerate(info["free_nodes"])}
    abits = int_to_bits(A, NBITS)
    bbits = int_to_bits(B, NBITS)
    cbits = [0] + [val[info["c"][i]] for i in range(1, NBITS + 1)]
    sbits = [val[info["s"][i]] for i in range(NBITS)]
    S = sum(sbits[i] << i for i in range(NBITS)) + (cbits[NBITS] << NBITS)
    H = exact_penalty(abits, bbits, sbits, cbits)
    return S, H


def solve(A: int, B: int, beta: float, schedule: SamplingSchedule,
          n_chains: int, key) -> dict:
    model, info = build_adder_model(beta)

    abits = int_to_bits(A, NBITS)
    bbits = int_to_bits(B, NBITS)
    clamp_vals = []
    for nd in info["clamped_block"]:
        if nd in info["a"]:
            clamp_vals.append(bool(abits[info["a"].index(nd)]))
        elif nd in info["b"]:
            clamp_vals.append(bool(bbits[info["b"].index(nd)]))
        else:                                    # c[0] carry-in = 0
            clamp_vals.append(False)
    clamp_state = [jnp.array(clamp_vals, dtype=bool)]

    program = IsingSamplingProgram(
        model, info["free_blocks"], [info["clamped_block"]]
    )
    observe = [Block(info["free_nodes"])]

    best_S, best_H = None, np.inf
    n_exact = 0
    n_total = 0
    for ch in range(n_chains):
        k = jax.random.fold_in(key, ch)
        k_init, k_samp = jax.random.split(k, 2)
        init_state = hinton_init(k_init, model, info["free_blocks"], ())
        samples = sample_states(
            k_samp, program, schedule, init_state, clamp_state, observe
        )
        arr = np.asarray(samples[0])             # (n_samples, n_free) bool
        for row in arr:
            S, H = decode(row, info, A, B)
            n_total += 1
            if H == 0:
                n_exact += 1
            if H < best_H:
                best_H, best_S = H, S

    return dict(
        A=A, B=B, true=A + B, pred=best_S, penalty=best_H,
        correct=(best_S == A + B and best_H == 0),
        ground_hit_rate=n_exact / n_total,
        ncolors=info["ncolors"],
    )


def main():
    BETA = 2.5
    SCHEDULE = SamplingSchedule(n_warmup=4000, n_samples=200,
                                steps_per_sample=4)
    N_CHAINS = 16

    tests = [
        (0, 0), (1, 1), (2, 3), (15, 15), (170, 85),
        (123, 200), (255, 1), (1, 255), (255, 255), (200, 199),
    ]
    rng = np.random.default_rng(0)
    tests += [(int(rng.integers(0, 256)), int(rng.integers(0, 256)))
              for _ in range(6)]

    key = jax.random.key(0)

    print("=" * 74)
    print("8-BIT BINARY ADDER  (THRML energy-based model, ground state = sum)")
    print("=" * 74)
    _, info0 = build_adder_model(BETA)
    print(f"free vars            : {len(info0['free_nodes'])} "
          f"(8 sum bits + 8 carries)")
    print(f"clamped inputs       : {len(info0['clamped_nodes'])} "
          f"(A[8] + B[8] + carry-in)")
    print(f"chromatic Gibbs cols : {info0['ncolors']}  "
          f"(graph has triangles -> not 2-colorable)")
    print(f"beta / schedule      : beta={BETA}, warmup="
          f"{SCHEDULE.n_warmup}, samples={SCHEDULE.n_samples}, "
          f"chains={N_CHAINS}")
    print()
    print(f"{'A':>5} {'B':>5} | {'A+B':>5} {'pred':>5} | {'pen':>4} "
          f"{'ground%':>8} | result")
    print("-" * 74)

    n_pass = 0
    for (A, B) in tests:
        r = solve(A, B, BETA, SCHEDULE, N_CHAINS,
                  jax.random.fold_in(key, A * 257 + B))
        ok = r["correct"]
        n_pass += ok
        print(f"{A:>5} {B:>5} | {r['true']:>5} {r['pred']:>5} | "
              f"{r['penalty']:>4} {r['ground_hit_rate'] * 100:>7.1f}% | "
              f"{'PASS' if ok else 'FAIL'}")

    print("-" * 74)
    print(f"Exact (best-of-chains) accuracy: {n_pass}/{len(tests)} "
          f"= {100 * n_pass / len(tests):.1f}%")
    print()
    print("Interpretation:")
    print("  - 'pen' is the exact integer penalty of the best sample; 0 means")
    print("    the sampler reached the ground state == correct arithmetic.")
    print("  - 'ground%' is the fraction of ALL raw samples that were exactly")
    print("    correct -- this is the honest single-shot reliability and")
    print("    illustrates why a sampler needs many shots / annealing for")
    print("    deterministic logic (the Wall D / mixing tradeoff).")
    raise SystemExit(0 if n_pass == len(tests) else 1)


if __name__ == "__main__":
    main()
