# Plan: A C Compiler for a TSU ("TSU-C")

**Status:** design plan. Grounded in `FEASIBILITY.md` (theory) and the working,
differentially-validated stack already in this repo (`stochastic_clock.py`,
`tsu_runtime.py`, `adder8.py`, `clock16.py`, `fibonacci.py`, `brainfuck.py`,
`brainfuck16.py`).

---

## 0. The thesis (read this first)

Compiling C "to a TSU" does **not** mean lowering a C program into one energy
function. `FEASIBILITY.md` proves that is impossible in general (Walls A–D +
unbounded computation), and `brainfuck16.py` already demonstrates the
*buildable* alternative end-to-end:

> A classical **host runtime** owns control flow, memory, and sequencing.
> Each **bounded arithmetic primitive** is one wide-basin EBM relaxation.
> The **stochastic clock** is the settle barrier between steps.

A C compiler is that exact pattern, scaled. Therefore the C compiler is, in
architecture terms, an **accelerator-offload compiler** (structurally like a
CUDA/TPU compiler): `C → IR → schedule → host program that calls a fixed ISA
of EBM kernels`. The hard, novel work is **not** the C front end (that is
standard). It is:

1. defining and **characterizing the TSU ISA** (the gadget library), and
2. accepting and enforcing a **bounded C subset** ("TSU-C").

### The single design principle that makes this conceivable

> **One deep gadget, everything else reduces to it or is trivial.**

The only EBM gadget with a deep carry chain is the **annealed adder**
(`tsu_runtime.AdderEngine`, already validated 8- and 16-bit). Every other C
operation is either (a) *trivial* on a TSU (bitwise/shift-by-constant: no
carry, wider basins than add), or (b) **host-sequenced loops of the adder /
comparison gadget** (multiply, divide, wide ints). We never build a
monolithic multiplier/divider EBM — that is exactly the Wall-A / factoring
hardness from `FEASIBILITY.md`. Depth is pushed into host sequencing, where
`brainfuck16.py` already proved loops of cheap relaxations work.

---

## 1. The supported subset: "TSU-C"

A general ISO C compiler is **provably impossible** here (unbounded memory /
recursion / loops ⇒ Walls A,B + halting). TSU-C is the bounded analogue of
the "limited Brainfuck" decision, made explicit:

| Area | Supported (v1) | Excluded (v1) | Why |
|---|---|---|---|
| Integer types | `_Bool, char, short, int` (8/16/32-bit) | `long long`>64, bitfields(v1) | width = adder depth = Wall A; 32/64 via *limbs* (§3) |
| Floating point | — | `float, double` | mantissa normalization is its own Wall-A problem; separate research |
| Arithmetic | `+ - * / % << >> & \| ^ ~` unary | — | all reduce to adder/bitwise gadgets (§4) |
| Control flow | `if/else while for do switch ?: && \|\|` | `goto` into blocks (v1) | host CFG; conditions via cmp gadget |
| Functions | params, returns, **bounded** recursion, fn pointers | varargs (v1) | host call stack, static depth bound = bounded computation |
| Memory | fixed RAM arena, pointers, arrays, `struct/union`, arena `malloc` | unbounded `malloc`, VLA | RAM is **host** state (like the BF tape); bounded ⇒ tractable |
| I/O | `putchar`/return value (no input) | `scanf`/file input | matches the "all commands but input" constraint we already use |
| UB | trap or define | rely on UB | determinism needed for differential validation |

Enforcement is a compiler pass that **rejects** out-of-subset constructs at
compile time (exactly as `compile_bf` rejects `,` and unbalanced brackets).
"Bounded" is real: max loop trip count, recursion depth, and RAM size are
compile-time/link-time constants; programs provably halt within `MAX_STEPS`.

---

## 2. Architecture: what runs where

The compiler partitions every C construct into **host** (classical, exact,
free) vs **TSU** (one EBM relaxation, ~seconds, sampled). This mirrors
`brainfuck16.py` (`< > [ ] .` host; `+ -` EBM) scaled to C:

| C construct | Executes on | Mechanism |
|---|---|---|
| Statement sequencing, CFG, branches, loops | **Host** | host program counter; cond from cmp gadget |
| Function call / return / stack frames | **Host** | bounded host call stack |
| Address/offset computation (`a[i]`, `p->f`) | **TSU** | adder gadget (then host array index) |
| Load / store / RAM | **Host** | RAM is a host array (bounds-checked) |
| `+ - ` and comparisons `== < <= > >= !=` | **TSU** | adder gadget (cmp = read carry/sign) |
| `& \| ^ ~`, shift by constant | **TSU (trivial)** | per-bit factors, no carry; or host rewire |
| shift by variable | **TSU** | mux network gadget (medium) |
| `* / %`, 32/64-bit ops | **Host-sequenced TSU** | loops of adder/cmp gadgets (§4) |
| Constants, dead code, identities | **Host (compile time)** | constant folding / strength reduction |

Crucial corollary: aggressive classical optimization is **load-bearing for
feasibility**, not just speed. Constant folding, strength reduction, common
sub-expression elimination, and **op fusion** directly reduce the number of
EBM relaxations — the same lesson as run-length folding in `brainfuck16`
(`+`×1000 → *one* solve). The optimizer is a correctness-of-runtime concern.

---

## 3. The TSU ISA (gadget library)

The compiler's true backend target. Each gadget has a measured reliability
class (from this repo's empirical results: 8-bit add ≈ 100% best-of-chains;
16-bit deep carries 2–65% single-op ground rate but exact via
anneal+best-of-chains; bitwise has *no* carry ⇒ strictly easier than add).

| Gadget | Arity / width | Reliability class | Implementation |
|---|---|---|---|
| `ADD/SUB w` | 2×w → w(+carry) | **medium** (Wall A ∝ w) | `tsu_runtime.AdderEngine`; SUB = add two's-complement (proven: BF `-` = add 0xFFFF) |
| `CMP w` | 2×w → {lt,eq,gt} | **medium** (≈ free given ADD) | compute `a-b`; read sign + zero + carry bits — *no new deep gadget* |
| `AND/OR/XOR/NOT w` | → w | **trivial** (1-local, no carry) | per-bit independent factors; widest basins |
| `SHL/SHR const` | → w | **free** | host wire/index remap at compile time |
| `SHL/SHR var` | → w | **medium** | log-depth mux gadget (or host-sequenced) |
| `MUX/SELECT` | sel,a,b → | **easy** | small EBM or host (used for `?:`, predication) |
| `MUL w` | 2×w → 2w | **host-sequenced** | shift-add: w iterations of `ADD` + `MUX` — *never monolithic* |
| `DIV/MOD w` | 2×w → w,w | **host-sequenced** | restoring division: w iterations of `SUB`+`CMP` |
| wide int (32/64) | limbs of 16 | **host-sequenced** | limb-wise `ADD` with carry-in chaining (host carries the carry) |

Design payoff: the **only** component that ever stresses Wall A is one
gadget, `ADD`, and it is already built, annealed, cached, and validated. The
entire C arithmetic surface is `ADD` + trivial bitwise + host loops. This is
the concrete realization of "one deep gadget, everything else reduces to it."

Each gadget ships with a **characterization datasheet**: ground-state hit
rate vs width/anneal schedule/chains, and a worst-case operand set — produced
by the same harness style as `clock16.py`'s carry-stress probes.

---

## 4. The hard operations, decomposed (not avoided)

- **Multiply** `z = x*y` (w-bit): host loop `for i in 0..w-1: if (y>>i)&1: z = ADD(z, x<<i)`. Each step is the *cheap* `ADD` gadget; `x<<i` is a free constant shift; `(y>>i)&1` is a host bit test. `w` relaxations, all wide-basin. Monolithic multiply (one EBM whose ground state is the product) is the *forbidden* construction — it is the factoring-hardness wall from `FEASIBILITY.md`.
- **Divide/modulo**: non-restoring division — `w` iterations of `SUB`+`CMP`(=ADD)+`MUX`. Same principle.
- **32/64-bit**: split into 16-bit limbs; `ADD` per limb with the carry-out fed (host) into the next limb's carry-in. We already expose carry I/O in the adder EBM (`c[0]` clamped, `c[nbits]` read). This is why v1 caps the *gadget* at 16-bit even though TSU-C supports `int`32: width is handled by host limb sequencing, never by a deeper EBM. Empirically this is mandatory — `brainfuck16` already shows 16-bit deep carries at 2–4% single-shot; a 32-bit monolith would fall off the cliff `FEASIBILITY.md` predicts.

---

## 5. Compiler pipeline

```
C source
  │  (1) Front end
  ├─ lex / parse (C11 subset grammar) → AST
  ├─ semantic analysis, type checking
  └─ TSU-C SUBSET CHECK  → hard compile error on excluded constructs
  │
  │  (2) Lowering to TIR (TSU IR: typed 3-address, SSA, explicit CFG)
  ├─ type legalization (→ 8/16-bit gadget widths; 32/64 → limb sequences)
  ├─ mem lowering (locals/arrays/structs → host RAM arena offsets)
  ├─ mul/div/shift-var → host-sequenced gadget loops
  └─ control flow → host CFG of basic blocks; conds → CMP gadget reads
  │
  │  (3) Optimization (feasibility-critical, not just speed)
  ├─ constant folding, strength reduction, CSE, copy/const propagation
  ├─ OP FUSION & range-narrowing  → minimize #EBM relaxations
  └─ gadget scheduling (batch independent ADDs; reuse cached programs)
  │
  │  (4) Backend
  ├─ emit HOST PROGRAM (Python/runtime bytecode): CFG + RAM + call stack
  ├─ emit GADGET CALL SCHEDULE (typed ISA ops, §3)
  └─ insert STOCHASTIC-CLOCK barriers (one tick / gadget = settle epoch)
  │
  ▼
Runnable artifact: host runtime + AdderEngine-backed gadget kernels
```

TIR is deliberately small (think a cut-down LLVM IR). The backend is *not* a
code generator for silicon; it emits a host control program plus a schedule
of `ENGINE.solve(...)`-style calls — the generalization of `run_tsu` in
`brainfuck16.py`.

---

## 6. Runtime & ABI

- **Memory**: one fixed host `bytearray` arena (size = link-time constant).
  Globals at low addresses; a bounded **stack** (frame = saved regs, params,
  locals, return slot); arena-`malloc` is a bump/free-list allocator over the
  same array. Every access bounds-checked → trap (deterministic, validatable).
- **Calling convention**: host-managed stack; args/locals are host RAM slots;
  return value in a host register slot. Recursion permitted up to a
  compile-time depth bound (overflow = trap). Function pointers = host indirect
  dispatch table.
- **Gadget dispatch**: a thin layer over `tsu_runtime.AdderEngine` (and the
  bitwise/mux engines). Inputs are clamped, outputs decoded, **penalty
  asserted 0**; on non-ground, escalate (more chains / hotter→colder anneal
  stages) then retry — `clock16.py` already proved annealing recovers deep
  carries. This escalation policy is the runtime's correctness guarantee
  against Wall D.
- **Clock barrier**: each gadget invocation consumes one stochastic-clock
  tick (`stochastic_clock.py`), the "relaxation epoch finished" synchronizer,
  exactly as in `brainfuck16.run_tsu`.

---

## 7. Validation methodology (the part that makes it credible)

Reuse and scale the differential harness that already validates
`brainfuck16.py`:

1. **Differential testing**: every TSU-C program is also compiled by a
   trusted reference (native `gcc`/`clang` or a clean C interpreter) on the
   same bounded inputs. Assert: identical return value, `putchar` stream, and
   final RAM-arena image.
2. **Per-gadget ground-state assertion**: every relaxation must hit penalty 0
   (exact arithmetic from sampling) or the run fails — the invariant we
   already enforce per `+`/`-`.
3. **Gadget datasheets**: characterization sweeps (width × anneal × chains ×
   worst-case operands), like `clock16.py`'s carry-stress probes.
4. **Conformance suite**, milestone-gated, one feature per test, growing like
   our `LIBRARY`: integer arithmetic & overflow, all operators, every
   control-flow form, pointers/arrays/structs, bounded recursion (e.g. an
   iterative *and* a depth-bounded recursive Fibonacci — we already have the
   iterative TSU Fibonacci), mul/div decomposition, limb-wise 32-bit.
5. **Determinism gate**: same program+input ⇒ identical observable result
   across runs (sampling is hidden behind the ground-state guarantee).

---

## 8. Milestones (each independently validated, like the existing scripts)

| M | Deliverable | Reuses |
|---|---|---|
| M0 | TSU-C grammar + subset checker + TIR + reference oracle harness | brainfuck16 validation pattern |
| M1 | Straight-line integer arithmetic (`+ - & \| ^ ~ <<c >>c`, cmp) → host+ADD | `tsu_runtime.AdderEngine` |
| M2 | Control flow (`if/while/for/switch/?:/&&/\|\|`) on host CFG | brainfuck16 control model |
| M3 | Memory: locals, arrays, pointers, structs over RAM arena | brainfuck16 tape → arena |
| M4 | Functions, calling convention, bounded recursion, fn pointers | new host stack |
| M5 | `* / %` and 32/64-bit via host-sequenced gadget loops | shift-add over ADD |
| M6 | Conformance suite + gadget datasheets + determinism gate | clock16 stress harness |
| M7 | Optimizer (folding/CSE/fusion) — cuts relaxation count | run-length-fold lesson |

Each milestone ends with a `pytest`-style differential suite that must pass
before the next — the same incremental, self-validating discipline used for
`stochastic_clock → adder8 → clock16 → fibonacci → brainfuck → brainfuck16`.

---

## 9. Risks, limits, honest verdict

- **Performance is demonstrator-only.** Every arithmetic op is ~seconds of
  sampling; a multiply is `w` of them; a real C workload is astronomically
  slow. This is a *correctness/architecture* artifact, exactly as
  `FEASIBILITY.md` flagged the 10,000× number as simulation-only. Do not
  promise practicality.
- **Wall A still bounds width.** Mitigation (limb sequencing) is mandatory,
  not optional; 32-bit monolithic gadgets are out of scope by design, and
  the gadget datasheets must *measure* the real width ceiling.
- **The subset is the product.** "We compile C" is an overclaim; "we compile
  a bounded, deterministic C subset to a host runtime + a validated EBM
  gadget ISA, differentially proven correct" is true and defensible.
- **Floats are a separate research line**, not a v1 slip.

**Verdict.** A general ISO C compiler for a TSU is impossible for the same
provable reasons as in `FEASIBILITY.md`. A **TSU-C cross-compiler** — bounded
C → host control runtime + a small, characterized, annealed-adder-centric
gadget ISA, validated differentially — is **buildable**, is a natural and
honest scaling of the already-working `brainfuck16.py`, and is the correct
thing to build. The whole risk reduces to one already-solved component (the
annealed adder) plus disciplined classical compiler engineering.

---

### Appendix: direct reuse map

| Need | Existing artifact |
|---|---|
| ADD/SUB/CMP gadget substrate | `tsu_runtime.AdderEngine` (cached, annealed, N-bit) |
| Anneal-to-recover deep carries | `clock16.py` schedule + best-of-chains |
| Settle barrier / sequencing | `stochastic_clock.py` (one-hot tick) |
| Host-sequenced execution model | `brainfuck16.run_tsu` (generalize to a CFG VM) |
| Subset rejection at compile time | `brainfuck16.compile_bf` (`,`/bracket errors) |
| Relaxation-count minimization | run-length folding in `brainfuck16.compile_bf` |
| Differential validation harness | `brainfuck16` ref-vs-TSU + tape compare |
| Bounded "tape" → RAM arena | `brainfuck16` 64-cell tape |
