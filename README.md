# TSU-C — a C compiler whose arithmetic runs on a thermodynamic computer

TSU-C is a small, hardened C compiler whose **arithmetic is executed as the
ground state of an energy-based model sampled on a (simulated) Thermodynamic
Sampling Unit (TSU)** — Extropic's probabilistic-bit substrate, via the
open-source [THRML](https://github.com/extropic-ai/thrml) library.

Every value-producing `+ - *` (and unary `-`) in a compiled program is not
computed by an ALU. It is obtained by letting a network of stochastic
"p-bits" relax into the minimum of an Ising energy function whose ground
state encodes the correct result — then read back. Control flow, memory and
sequencing stay on a classical host; only the numeric kernel is thermodynamic.

> **Status:** working and differentially validated. Not a practical compiler
> (every operation is seconds of sampling) — it is a *correctness and
> architecture demonstrator* for thermodynamic computing.

---

## The idea in one paragraph

A TSU is a sampler: it draws from `P(x) ∝ exp(−βE(x))`. You cannot lower an
arbitrary program into one giant energy landscape — that is provably
intractable (see [`FEASIBILITY.md`](FEASIBILITY.md): mixing-time / analog-
precision / embedding / sample-vs-decide walls, plus the halting problem).
What *does* work is **host-sequenced time-slicing**: a classical control
program runs the CFG, memory and call stack, and offloads each *bounded
arithmetic primitive* to one wide-basin EBM relaxation. TSU-C is the
realization of that design (specified in
[`PLAN_C_COMPILER.md`](PLAN_C_COMPILER.md)) for the C language.

## Architecture

```
C source
  │  lexer  → recursive-descent parser → typed AST
  │  static analysis  (scoping, types, arity, subset enforcement)
  │  constant folding  (all-constant subexpressions → no EBM ops)
  ▼
two interpreters over the SAME folded AST, identical control flow:
  • reference : exact host integers/fixed-point  (the oracle)
  • TSU       : every  + - *  is an EBM ground state on a 16-bit
                annealed adder, composed to 32/64-bit via the
                16-bit-LIMB technique, with escalate-and-retry
```

- **One deep gadget.** The only thermodynamic primitive is a β-annealed
  16-bit adder (`tsu_runtime.AdderEngine`, built once, cached). Subtraction =
  add of two's-complement; multiply = shift-add over the adder; wider widths
  = 16-bit limbs with host carry chaining. Comparisons, bitwise ops, shifts
  and scaling are host (the "free" class).
- **Determinism guarantee.** Sampling is probabilistic, so a relaxation can
  miss the ground state (Wall D in `FEASIBILITY.md`). The runtime detects a
  non-ground result (penalty ≠ 0) and **escalates-and-retries** (fresh keys,
  then a stronger 10-chain / 4-stage recovery engine) until exact, or traps.
  Results are therefore deterministic and exact despite the stochastic
  substrate; recoveries are reported transparently.
- **Emergent clock.** Instruction dispatch is paced by a pbit *stochastic
  clock* (`stochastic_clock.py`) — one one-hot tick per EBM op acts as the
  "relaxation epoch finished" barrier.

## Supported C ("TSU-C")

A general ISO C compiler is impossible on this substrate (Walls A–D +
unbounded computation). TSU-C is the **bounded** subset, enforced with
line-numbered compile errors.

**Numeric model (all fixed-point — IEEE-754 is the documented Wall-A line):**

| type     | representation        | width  | range / step                |
|----------|-----------------------|--------|-----------------------------|
| `int`    | two's complement      | 32-bit | ±2.1 × 10⁹                  |
| `long`   | two's complement      | 64-bit | ±9.2 × 10¹⁸                 |
| `float`  | Q16.16 fixed-point    | 32-bit | ±3.3 × 10⁴, step ~1.5e-5    |
| `double` | Q32.32 fixed-point    | 64-bit | ±2.1 × 10⁹, step ~2.3e-10   |

Promotion lattice `int < long < float < double`; implicit conversions,
`(type)` casts, C literal typing (`3.14` is `double`, `3.14f` is `float`,
`5000000000L` / oversized constants are `long`).

**Supported:** all integer/float operators (`+ - * / % & | ^ ~ << >>`,
relational, `&& || !`, `?:`, compound assignment); `if/else`, `while`,
`do/while`, `for`, `switch`-free structured control, `break`/`continue`;
functions with parameters, returns, bounded recursion, function-free
prototypes; globals, lexical block scoping; `putchar` output.

**Rejected at compile time:** pointers, arrays, `struct`/`union`, `char`
variables, `float`/IEEE semantics, runtime input (`scanf`), `goto`, string
literals, unbounded loops/recursion, and `short`/`unsigned`/`typedef`/… —
plus the usual malformed-input diagnostics (unbalanced braces, undeclared
variables, unknown calls, arity mismatch, bad literals, unterminated
comments, etc.). Deeply nested input fails gracefully, never with a Python
traceback.

## Honest limitations

- **Fixed-point, not IEEE-754.** No NaN/Inf/denormals/rounding modes; `float`
  has limited range/precision, `double` simply wider. Conversions truncate
  toward zero (deterministic, documented).
- **Division is host-evaluated** (all domains), reported as `div_host`. Its
  EBM lowering is the subtract gadget already validated for integers; it is
  not re-run on the substrate for runtime-budget reasons.
- **Demonstrator performance.** Each EBM op is ~seconds of sampling; a real
  workload would be astronomically slow. The artifact proves *correctness and
  architecture*, exactly as `FEASIBILITY.md` framed the 10,000× efficiency
  claim as simulation-only.
- **Bounded computation only.** Loop trip counts, recursion depth and step
  budget are capped; programs provably halt. No runtime input.

## Repository layout

| file | role |
|------|------|
| **`tsuc.py`**            | the TSU-C compiler (lexer, parser, types, folding, EBM/host interpreters) |
| **`validate_tsuc.py`**   | integer/`long` EBM-backed differential suite vs the reference oracle |
| **`validate_float.py`**  | `float`/`double` reference battery + EBM-backed differential subset |
| **`harden_tsuc.py`**     | fast (no-EBM) robustness suite: malformed input, scoping, limits, regression |
| `tsu_runtime.py`         | cached β-annealed N-bit adder engine (the one deep gadget) |
| `stochastic_clock.py`    | pbit stochastic clock (emergent dispatch/settle barrier) |
| `adder8.py`, `clock16.py`| lower-level EBM adder / annealed-clock demonstrators |
| `brainfuck.py`, `brainfuck16.py` | earlier 8/16-bit Brainfuck compilers (same architecture, simpler language) |
| `fibonacci.py`           | clock + adder composition demo |
| `FEASIBILITY.md`         | theory: what a TSU can and cannot run, and why |
| `PLAN_C_COMPILER.md`     | the full C-compiler design this implements |
| `paper_outline.md`       | write-up outline |

## Quick start

Requires the `.venv` with `thrml` + `jax` (already provisioned).

```bash
# fast — front end, semantics, robustness; NO sampling (seconds)
.venv/Scripts/python.exe harden_tsuc.py

# EBM-backed: integer & long arithmetic on the TSU (minutes)
.venv/Scripts/python.exe validate_tsuc.py

# EBM-backed: float & double on the TSU (minutes)
.venv/Scripts/python.exe validate_float.py
```

Each suite is **differential**: the EBM-backed run must match the exact
reference oracle bit-for-bit, every relaxation must reach ground state, and
the stochastic clock must advance monotonically — otherwise the suite fails.

### Minimal example

```python
from tsuc import compile_src, run_reference

src = """
long fact(int n){ if (n <= 1) return 1; return n * fact(n-1); }
long main(){ return fact(15); }
"""
print(run_reference(compile_src(src)))   # (1307674368000, b'')
```

Run it on the thermodynamic substrate instead (every `*`/`-` an EBM ground
state) with `run_tsu(prog, key, tick_steps)` — see `validate_tsuc.py`.

## Validation status

All three suites pass:

- **`harden_tsuc.py`** — every malformed/out-of-subset program rejected
  gracefully with line numbers; ~60 correctness programs exact (scoping,
  overflow/wrap, signed `%`//, short-circuit, recursion, hex/char/suffix
  literals, casts, `int`-32 range, `long`-64 incl. `15! = 1.3e12`,
  `float`/`double`); pathological programs stopped; full regression clean.
- **`validate_tsuc.py`** — 16/16 programs; integer & `long` `+ − *` executed
  as EBM ground states (32-bit = 2 limbs, 64-bit = 4); 141 relaxations,
  escalate-and-retry recovered every non-ground sample to an exact result.
- **`validate_float.py`** — 20 reference + 9 EBM-backed `float`/`double`
  programs; Q16.16 / Q32.32 arithmetic exact on the substrate.

## Further reading

- [`FEASIBILITY.md`](FEASIBILITY.md) — why a TSU can't run arbitrary programs
  in one energy landscape, and the tractable/intractable map.
- [`PLAN_C_COMPILER.md`](PLAN_C_COMPILER.md) — the full compiler design.
- Extropic *THRML* and the architecture paper *“An efficient probabilistic
  hardware architecture for diffusion-like models”* (arXiv:2510.23972).
