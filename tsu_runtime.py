"""
Reusable TSU arithmetic runtime: a cached, beta-annealed N-bit adder engine.

The earlier clock16.anneal_solve rebuilt the entire EBM (nodes, QUBO,
coloring, sampling program) for every chain x stage x operation. For a whole
Brainfuck program that is thousands of redundant builds.

`AdderEngine` builds the per-beta-stage models and sampling programs ONCE in
the constructor and reuses them for every solve -- only the clamped input
vector (A, B) changes per operation. This is the "compile the datapath once,
execute it many times" move, and it makes a 16-bit library feasible.

Semantics are identical to clock16.anneal_solve: simulated annealing across
rising-beta stages (chain state carried forward), best-of-chains readout,
ground state of E(x) ∝ (a+b+c - s - 2c')^2 == correct addition.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def _build(nbits: int, beta: float):
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
            weights.append(-j)

    model = IsingEBM(nodes, edges, jnp.array(-h), jnp.array(weights),
                     jnp.array(beta))

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
    free_blocks = [Block([nd for nd in free_nodes if color[nd] == col])
                   for col in range(ncolors)]
    clamped_block = Block(clamped_nodes)

    # clamp plan: for each clamp slot, ('A', i) / ('B', i) / ('Z', 0)
    plan = []
    aset = {nd: i for i, nd in enumerate(a)}
    bset = {nd: i for i, nd in enumerate(b)}
    for nd in clamped_block:
        if nd in aset:
            plan.append(("A", aset[nd]))
        elif nd in bset:
            plan.append(("B", bset[nd]))
        else:
            plan.append(("Z", 0))

    pos = {nd: k for k, nd in enumerate(free_nodes)}
    info = dict(s=s, c=c, free_nodes=free_nodes, free_blocks=free_blocks,
                clamped_block=clamped_block, ncolors=ncolors, plan=plan,
                pos=pos, model=model)
    program = IsingSamplingProgram(model, free_blocks, [clamped_block])
    observe = [Block(free_nodes)]
    return info, program, observe


class AdderEngine:
    """Cached beta-annealed N-bit adder. Build once, solve many."""

    def __init__(self, nbits: int, stages, n_chains: int):
        self.nbits = nbits
        self.stages = stages
        self.n_chains = n_chains
        self._cache = [(_build(nbits, beta), nw) for (beta, nw) in stages]
        self.ncolors = self._cache[0][0][0]["ncolors"]
        self.n_free = len(self._cache[0][0][0]["free_nodes"])

    def _clamp_state(self, info, A: int, B: int):
        abits = [(A >> i) & 1 for i in range(self.nbits)]
        bbits = [(B >> i) & 1 for i in range(self.nbits)]
        vec = []
        for (tag, i) in info["plan"]:
            if tag == "A":
                vec.append(bool(abits[i]))
            elif tag == "B":
                vec.append(bool(bbits[i]))
            else:
                vec.append(False)
        return [jnp.array(vec, dtype=bool)]

    def _flat_to_color_init(self, info, flat_bits: np.ndarray):
        pos = info["pos"]
        return [jnp.array([bool(flat_bits[pos[nd]]) for nd in blk],
                          dtype=bool)
                for blk in info["free_blocks"]]

    def solve(self, A: int, B: int, key) -> dict:
        nb = self.nbits
        abits = [(A >> i) & 1 for i in range(nb)]
        bbits = [(B >> i) & 1 for i in range(nb)]
        best_val, best_H, n_exact, n_total = None, np.inf, 0, 0

        for ch in range(self.n_chains):
            k = jax.random.fold_in(key, ch)
            (info0, _, _), _ = self._cache[0]
            k_i, k = jax.random.split(k, 2)
            state = hinton_init(k_i, info0["model"], info0["free_blocks"], ())

            for sidx, ((info, program, observe), nw) in enumerate(
                    self._cache):
                last = sidx == len(self._cache) - 1
                clamp_state = self._clamp_state(info, A, B)
                sch = SamplingSchedule(
                    n_warmup=nw,
                    n_samples=(80 if last else 1),
                    steps_per_sample=(4 if last else 1),
                )
                k_s, k = jax.random.split(k, 2)
                out = np.asarray(sample_states(
                    k_s, program, sch, state, clamp_state, observe)[0])
                state = self._flat_to_color_init(info, out[-1])

                if last:
                    for row in out:
                        val = {nd: int(bool(row[i]))
                               for i, nd in enumerate(info["free_nodes"])}
                        cbits = [0] + [val[info["c"][i]]
                                       for i in range(1, nb + 1)]
                        sbits = [val[info["s"][i]] for i in range(nb)]
                        S = (sum(sbits[i] << i for i in range(nb))
                             + (cbits[nb] << nb))
                        H = 0
                        for i in range(nb):
                            r = (abits[i] + bbits[i] + cbits[i]
                                 - sbits[i] - 2 * cbits[i + 1])
                            H += r * r
                        n_total += 1
                        n_exact += (H == 0)
                        if H < best_H:
                            best_H, best_val = H, S

        return dict(pred=best_val, penalty=best_H,
                    ground_hit_rate=n_exact / n_total,
                    correct=(best_val == A + B and best_H == 0))
