# Feasibility Map: A General-Purpose Compiler for an Extropic TSU

**Question:** Can a compiler take an arbitrary program and run it on a Thermodynamic Sampling Unit (TSU)?
**Short answer:** Not in general — and the obstruction is information-theoretic and physical, not engineering. A *restricted constraint compiler* for a well-defined class is buildable. This document works the math and draws the tractable/intractable line on Extropic's **actual** hardware primitives.

---

## 0. The hardware model we are compiling to (from arXiv:2510.23972)

The TSU is **not** a general factor machine. It is precisely:

> A sampler from the Gibbs distribution `P(x) ∝ exp(−β E(x))` of a **2-local Ising/Boltzmann energy**
> `E(x) = −β ( Σ_{i≠j} xᵢ Jᵢⱼ xⱼ + Σᵢ hᵢ xᵢ )`,
> over **binary** variables on a **sparse, bipartite (2-colorable), locally-connected lattice** (paper's example: 70×70 grid, fan-in ≈ 12 neighbors), with **analog** weights/biases (sigmoidal bias circuit + resistor-network current summation — *no high-bit-depth digital precision*), updated by **block (chromatic) Gibbs sampling in 2 color phases**, optionally stacked/annealed as a **Denoising Thermodynamic Model (DTM)**.

Five hard constraints fall out of this and drive everything below:

| # | Constraint | Consequence for a compiler |
|---|---|---|
| C1 | **2-local only** (pairwise factors) | Every higher-arity logical/arithmetic relation must be reduced to pairwise with **ancilla variables** → variable blow-up + harder landscape. |
| C2 | **Sparse local bipartite topology**, fan-in ≈ 12 | Arbitrary factor graphs must be **minor-embedded** into a fixed lattice → chains, chain-strength tuning, coupling attenuation ∝ 1/(chain length). Authors explicitly concede wire-topology ≠ problem-topology. |
| C3 | **Analog couplings** (no stated bit-depth; resistor + thermal noise → ~4–8 *effective* bits) | The ratio (largest enforceable penalty)/(smallest meaningful coupling) is **bounded**. This is a hard dynamic-range ceiling. |
| C4 | **Block Gibbs = MCMC** | Output quality is governed by **mixing time ≈ 1/spectral gap**. Glassy landscapes ⇒ exponential time. |
| C5 | **It samples; it does not compute a value** | Returns a *draw* from a distribution, not a guaranteed unique answer. Determinism must be *engineered* against C3/C4. |

---

## 1. The model of computation, stated precisely

A TSU program is a triple `(G, J, h)` plus `(β, schedule)`, defining a Boltzmann distribution. "Running a program" means: clamp input variables, sample, decode output variables from the returned configuration(s). Computation = **search for low-energy configurations of a 2-local Ising model by chromatic Gibbs MCMC**.

This places the entire enterprise inside a body of theory that already has sharp answers.

---

## 2. Is it universal? (The "yes, in principle")

**Yes, for *bounded* computation — this part is solid.**

1. **Ground-state hardness ⇒ expressiveness.** Finding the ground state of a 2-local Ising model on a non-planar graph is **NP-hard** (Barahona, *J. Phys. A* 1982). Anything in NP — hence any bounded verifier — is encodable as such an energy function (Lucas, *Front. Physics* 2014, "Ising formulations of many NP problems").
2. **Bounded execution traces are encodable.** The Cook–Levin tableau encodes a time-`T`, space-`S` Turing/circuit computation as a SAT instance whose satisfying assignment **is** the full execution trace; map SAT→Ising via Tseitin gadgets. The quantum analog is the **Feynman–Kitaev clock Hamiltonian** (Kitaev 1999; Aharonov et al., adiabatic universality 2007). **This is exactly the user's §6–§12 "emergent clock + program phases + completion" idea** — it is a real, known construction, not a dead end conceptually.

So a compiler *can* in principle lower any **bounded, loop-unrolled** program to `(G, J, h)`. The theory blesses the *expressiveness*. It is the *efficiency and reliability* that collapse — next section.

---

## 3. Why "arbitrary program" fails in practice (the four walls, with the math)

### Wall A — Mixing time (C4). *The dominant wall.*
Block Gibbs is a Markov chain with transition matrix `K`. Time to get a usable sample is the relaxation time
`τ_rel ≈ 1 / gap(K)`, where `gap` is the spectral gap.
For energy landscapes encoding **deep deterministic logic**, the landscape is a **spin glass**: many deep, frustrated minima separated by `Θ(N)`-height barriers. For such landscapes the gap closes **exponentially**: `gap(K) ~ e^{−cN}` ⇒ `τ_rel ~ e^{cN}`. This is the same wall that prevents simulated/quantum annealing from solving worst-case instances, and the formal content of the user's "False Convergence." It is **not** fixable by a better schedule — it is a property of the encoded landscape.

### Wall B — Analog dynamic range (C3). *A hardware impossibility, not just slowness.*
To make a logical constraint *binding*, its penalty must dominate every competing term and thermal noise:
`J_penalty ≫ kT` **and** `J_penalty ≫ Σ (other incident couplings)`.
Chained constraints (carry chains, a Feynman–Kitaev clock, any depth-`D` pipeline) require the enforced energy scale to separate **correct vs. nearest-incorrect** by a gap `Δ` that must remain resolvable while *also* spanning `D` nested constraints. The required dynamic range grows like `Ω(D)` (often `Ω(2^{carry-depth})` for ripple arithmetic). With analog precision of `b` effective bits, you need `log₂(range) ≤ b`. With `b ≈ 4–8`, **deep encodings silently lose their ground-state guarantee**: the "correct answer = global minimum" property becomes false on the physical device even though it holds in the idealized math. No annealing schedule rescues a constraint the hardware cannot represent.

### Wall C — Embedding overhead (C1 + C2).
Arbitrary program → arbitrary factor graph. Two compulsory, lossy lowerings:
- **Higher-order → pairwise** (Boros–Hammer; Ishikawa 2011): each ≥3-variable factor spawns ancillas; variable count and frustration rise.
- **Minor-embedding** the result into the degree-≈12 bipartite lattice (Choi 2008; D-Wave `minorminer`): a logical node of degree `Δ` becomes a **chain** of physical pbits; effective logical coupling attenuates ≈ `1/ℓ` with chain length `ℓ`, and chain-break post-processing trades correctness against mixing. Treewidth-`w` problems need lattice regions of size `Ω(2^w)` to embed cleanly. The authors themselves flag this exact mismatch.

### Wall D — Sampling ≠ deciding (C5).
Even with infinite time and precision: a sampler returns configuration `x` with probability `∝ e^{−βE(x)}`. For a computation with a **unique** correct output among `2ⁿ` (a hash, a bijection, exact factoring), the correct state's probability share is `e^{−βΔ}/Z`. Reliable recovery needs `β → ∞`, which **freezes the chain** (Wall A worsens as `β` rises). This is the **annealing dilemma**, formally:

> For hard instances you can have a **fast-mixing** chain **or** a chain **concentrated on the answer**, *not both*.

Walls A–D are independent and each is individually fatal for the *general* case. They do **not** all fire for *every* program — that gap is the entire opportunity, and the subject of §4.

---

## 4. The tractability classification (the actual map)

The single predictive axis is **landscape ruggedness / solution-basin width**: how the measure of the acceptable-answer set scales, and how rugged the path to it is.

### ✅ TRACTABLE — native fit (sampling *is* the algorithm; wide basins)
- **Sampling from a learned EBM / Boltzmann machine / DTM.** This is literally what the silicon was built for. Zero impedance mismatch.
- **Probabilistic inference**: marginals / MAP / posterior sampling on bounded-treewidth or loosely-coupled factor graphs; Bayesian models.
- **Associative recall, pattern completion, denoising** (Hopfield-like): correct answer sits in a *broad* attractor by construction.
- **Combinatorial optimization as a heuristic** where *any good* solution suffices and the instance is not adversarial: Max-Cut, generic QUBO, Ising spin models, soft scheduling/routing — competitive with classical metaheuristics, *not* an exact solver.
- **Under-constrained / many-solution CSP & SAT** (easy regime), soft constraint satisfaction.

### ⚠️ MARGINAL — works small, degrades exponentially
- **Fixed-width invertible arithmetic** (add/multiply run forward *or* backward): demonstrated for small widths by **Camsari, Faria, Sutton & Datta, *Phys. Rev. X* 7, 031014 (2017)** ("Stochastic p-bits for invertible logic"). Degrades fast with bit-width; *inverse multiply* (factoring) hits Walls A+D quickly.
- **Shallow bounded logic with few solutions**; thin-but-not-unique feasibility problems.

### ⛔ PROVABLY HARD / INFEASIBLE — do not target
- **Deep deterministic pipelines with a unique answer**: cryptographic hashes/ciphers, exact integer factoring, bijective transforms — isolated needle ⇒ Walls A+D.
- **High-precision exact arithmetic over many chained ops** — Wall B (analog range) ⇒ ground-state guarantee fails on-device.
- **Unbounded loops / recursion / data-dependent halting** — *not encodable at all*; requires unbounded unrolling. Outside the model.
- **Worst-case NP-hard instances expecting the exact optimum** — no asymptotic edge over classical heuristics; usually *worse* after embedding overhead (Wall C).
- **Anything requiring guaranteed determinism, exactness, or bit-reproducibility** — contradicts C5.
- **The single-landscape Feynman–Kitaev "emergent clock" of §6–§12** — provably in this column: clock-Hamiltonian gap closes as `O(1/T²)` *before* disorder, and a deep history-tableau exceeds the analog dynamic range (Wall B). Sound as theory; not viable as a v1 execution model on this hardware.

---

## 5. A quantitative feasibility predicate (use before compiling anything)

Given a candidate program, estimate:

- `n` = #binary vars after loop-unroll + higher-order→pairwise + embedding ancillas
- `D` = logical depth (longest dependency chain of constraints)
- `Δ` = required energy gap, in units of `kT`, between the correct state and the nearest incorrect one (reliability needs `Δ ≳ 3–5 kT`)
- `b` = effective analog precision bits of the device (assume `b ≈ 4–8` until measured)
- `ℓ` = max minor-embedding chain length in the degree-≈12 bipartite lattice
- ruggedness flag `R` = {wide-basin / ferromagnetic-like} vs. {unique-deep-minimum / glassy}

**Compile only if all hold:**
1. `R = wide-basin` (or an *approximate* answer is acceptable), **and**
2. `log₂(dynamic range needed) ≈ log₂(Δ · D) ≤ b` (Wall B), **and**
3. `ℓ` small (single digits) so `1/ℓ` coupling attenuation stays above noise (Wall C), **and**
4. estimated `τ_rel` (proxy: frustration index / barrier height) is polynomial, not `e^{Θ(n)}` (Wall A).

Fail any ⇒ the program belongs in §4's ⛔ column; do not lower it — return a compiler error, not a slow binary.

---

## 6. Consequences for the compiler design (carry-forward)

1. **Target the ✅ class only.** The product is a *constraint/inference compiler*, not a CPU compiler. Frame it as such publicly to avoid the "arbitrary program" overclaim.
2. **Host-sequenced time-slicing, not one giant landscape.** Classical host runs control flow; offloads *shallow, wide-basin* sampling kernels per slice; carries state between relaxations. This keeps `D` (and thus the Wall B dynamic-range demand) bounded per call and sidesteps the §6–§12 clock entirely.
3. **A gadget library with *measured* gaps.** Each primitive lowered to a pairwise penalty block whose `Δ`, ancilla count, embedding footprint, and empirical mixing time are *characterized on THRML/silicon*, not assumed.
4. **The feasibility predicate (§5) is a compiler pass**, run before code generation, emitting a hard diagnostic when a construct provably lands in ⛔.
5. **Defer the emergent stochastic clock** to a research track; it is the single component most likely to fail and the easiest to replace with host sequencing.

---

## 7. Verdict

- *Expressiveness:* a bounded program **can** be encoded (Cook–Levin / Feynman–Kitaev / Lucas). ✔
- *General-purpose efficiency & reliability:* **impossible** on this substrate — Walls A (mixing), B (analog range), C (embedding), D (sample≠decide) are independent and each fatal for the deterministic/unique-answer general case. ✘
- *Restricted constraint/inference compiler for the ✅ class:* **feasible and valuable**, and it is exactly what the hardware was physically built to do. ✔

Recommended next step: turn §5 into a runnable analyzer and §4-✅ into a measured gadget library on the existing THRML setup (`sample.py` already runs a 5-spin Ising chain — the right starting substrate).

---

### Primary sources
- Jelinčič, Lockwood, Garlapati, Verdon, McCourt — *An efficient probabilistic hardware architecture for diffusion-like models*, arXiv:2510.23972 (2025). **(The TSU hardware model.)**
- Barahona — *On the computational complexity of Ising spin glass models*, J. Phys. A 15 (1982). **(Ground-state NP-hardness.)**
- Lucas — *Ising formulations of many NP problems*, Front. Physics 2 (2014). **(Problem→Ising encodings.)**
- Kitaev, Shen, Vyalyi (2002); Aharonov et al. — *Adiabatic quantum computation is equivalent to standard QC*, FOCS 2004 / SICOMP 2007. **(Feynman–Kitaev clock = the §6–§12 idea + its gap scaling.)**
- Camsari, Faria, Sutton, Datta — *Stochastic p-bits for invertible logic*, Phys. Rev. X 7, 031014 (2017). **(Invertible arithmetic + its breakdown.)**
- Choi — *Minor-embedding in adiabatic quantum computation*, Quantum Inf. Process. 7 (2008); D-Wave `minorminer`. **(The embedding pass + its overhead.)**
- Ishikawa — *Transformation of general higher-order MRFs to pairwise*, IEEE TPAMI 33 (2011). **(Higher-order → 2-local reduction.)**
