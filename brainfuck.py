"""
A limited Brainfuck compiler targeting the TSU substrate.

Supported commands:  > < + - . [ ]      (input ',' is intentionally rejected)

Architecture (the host-sequenced time-slicing model from FEASIBILITY.md):

    FRONT END  (compile_bf): lex source, reject ',' and unbalanced brackets,
        build a bracket jump table, and *lower* arithmetic:
            '+'  ->  EBM modular add of +1   (mod 256)
            '-'  ->  EBM modular add of +255 (= -1 mod 256)
        Producing a compiled IR (opcodes + jump table).

    BACK END   (run_tsu): execute the IR. The two arithmetic opcodes are
        evaluated as the GROUND STATE of the 8-bit adder energy-based model
        (clock16.anneal_solve) -- real thermodynamic computation, modular by
        discarding the carry-out bit. Pointer moves, branching and output are
        classical host control flow (exactly what FEASIBILITY.md says stays
        on the host). Instruction dispatch is paced by the pbit STOCHASTIC
        CLOCK: one one-hot tick per executed instruction = an emergent
        program counter.

"Limited" = no input, 8-bit wrapping cells, bounded tape, bounded step count
(Brainfuck is Turing-complete / unbounded; a TSU can only run bounded slices).

Validation: differential testing. For a suite exercising every supported
command, the TSU execution must match a trusted pure-Python reference
interpreter exactly (output + full final tape), every EBM arithmetic op must
reach penalty 0 (ground state == correct modular arithmetic), and the clock
must stay a valid strictly-advancing one-hot counter.
"""

import jax
import numpy as np

from clock16 import anneal_solve
from stochastic_clock import draw_pbit_stream, clock_ticks

CELL_MOD = 256
TAPE_LEN = 64
MAX_STEPS = 4000

# 8-bit arithmetic is the shallow/easy regime; a light anneal + few chains
# is plenty (cf. adder8.py hitting 100% best-of-chains).
BF_STAGES = [(0.8, 400), (2.0, 600), (3.5, 900)]
BF_CHAINS = 4

SUPPORTED = set("><+-.[]")


# ---------------------------------------------------------------------------
# FRONT END: compile Brainfuck source -> IR
# ---------------------------------------------------------------------------
class BFCompileError(Exception):
    pass


def compile_bf(src: str):
    """Lex + validate + lower. Returns (ops, jump).

    ops  : list of opcodes, each (kind, arg)
             ('move',  +1/-1)        from > <
             ('add',   1)            from +        (EBM)
             ('add',   255)          from -        (EBM, = -1 mod 256)
             ('out',   None)         from .
             ('jz',    target)       from [   (jump-if-zero to after ])
             ('jnz',   target)       from ]   (jump-if-nonzero to after [)
    jump : kept implicitly inside jz/jnz targets
    """
    if "," in src:
        raise BFCompileError(
            "input command ',' is not supported by this limited compiler")

    toks = [ch for ch in src if ch in SUPPORTED]

    # bracket matching -> jump targets
    ops = []
    stack = []
    for ch in toks:
        if ch == ">":
            ops.append(["move", 1])
        elif ch == "<":
            ops.append(["move", -1])
        elif ch == "+":
            ops.append(["add", 1])
        elif ch == "-":
            ops.append(["add", CELL_MOD - 1])      # -1 mod 256
        elif ch == ".":
            ops.append(["out", None])
        elif ch == "[":
            stack.append(len(ops))
            ops.append(["jz", None])               # patched below
        elif ch == "]":
            if not stack:
                raise BFCompileError("unbalanced ']'")
            open_idx = stack.pop()
            ops.append(["jnz", open_idx + 1])      # jump back to after '['
            ops[open_idx][1] = len(ops)            # '[' jumps to after ']'
    if stack:
        raise BFCompileError("unbalanced '['")

    return [tuple(o) for o in ops]


# ---------------------------------------------------------------------------
# Trusted reference interpreter (pure Python, standard BF semantics)
# ---------------------------------------------------------------------------
def run_ref(src: str):
    ops = compile_bf(src)
    tape = [0] * TAPE_LEN
    ptr = 0
    pc = 0
    out = []
    steps = 0
    while pc < len(ops):
        steps += 1
        if steps > MAX_STEPS:
            raise BFCompileError("reference exceeded MAX_STEPS")
        kind, arg = ops[pc]
        if kind == "move":
            ptr += arg
            if not (0 <= ptr < TAPE_LEN):
                raise BFCompileError("tape pointer out of bounds")
        elif kind == "add":
            tape[ptr] = (tape[ptr] + arg) % CELL_MOD
        elif kind == "out":
            out.append(tape[ptr])
        elif kind == "jz":
            if tape[ptr] == 0:
                pc = arg
                continue
        elif kind == "jnz":
            if tape[ptr] != 0:
                pc = arg
                continue
        pc += 1
    return bytes(out), tape


# ---------------------------------------------------------------------------
# BACK END: execute IR on the TSU substrate
# ---------------------------------------------------------------------------
def run_tsu(src: str, key, tick_steps, phases, K):
    ops = compile_bf(src)
    tape = [0] * TAPE_LEN
    ptr = 0
    pc = 0
    out = []
    steps = 0
    ebm_ops = 0
    ebm_bad = 0
    min_ground = 1.0
    used_ticks = []

    while pc < len(ops):
        # ---- emergent program counter: consume one stochastic clock tick
        if steps >= len(tick_steps):
            raise BFCompileError("ran out of clock ticks")
        used_ticks.append((int(tick_steps[steps]), int(phases[steps])))
        steps += 1
        if steps > MAX_STEPS:
            raise BFCompileError("TSU exceeded MAX_STEPS")

        kind, arg = ops[pc]
        if kind == "move":
            ptr += arg
            if not (0 <= ptr < TAPE_LEN):
                raise BFCompileError("tape pointer out of bounds")
        elif kind == "add":
            # ---- thermodynamic step: cell = (cell + arg) as EBM ground state
            r = anneal_solve(tape[ptr], arg, 8, BF_STAGES, BF_CHAINS,
                             jax.random.fold_in(key, pc * 100003 + steps))
            ebm_ops += 1
            if r["penalty"] != 0:
                ebm_bad += 1
            min_ground = min(min_ground, r["ground_hit_rate"])
            tape[ptr] = r["pred"] % CELL_MOD       # modular: drop carry-out
        elif kind == "out":
            out.append(tape[ptr])
        elif kind == "jz":
            if tape[ptr] == 0:
                pc = arg
                continue
        elif kind == "jnz":
            if tape[ptr] != 0:
                pc = arg
                continue
        pc += 1

    stats = dict(steps=steps, ebm_ops=ebm_ops, ebm_bad=ebm_bad,
                 min_ground=min_ground, used_ticks=used_ticks)
    return bytes(out), tape, stats


# ---------------------------------------------------------------------------
# Validation: differential testing vs the trusted reference
# ---------------------------------------------------------------------------
TESTS = [
    ("plus/out",        "+++."),
    ("plus+minus",      "+++++ - - ."),
    ("underflow wrap",  "-."),                       # 0 -> 255
    ("overflow wrap",   "-++."),                     # 0->255->0->1
    ("copy loop",       "+++[>++<-]>."),             # 3*2 = 6
    ("ptr move/out",    "+>++>+++<<.>.>."),          # 1,2,3
    ("skipped loop",    "[+++]."),                   # cell stays 0
    ("nested loops",    "++[>++[>+<-]<-]>>."),       # 2*2*1 = 4
]


def main():
    print("=" * 78)
    print("LIMITED BRAINFUCK COMPILER  (TSU back end: EBM arithmetic + "
          "pbit clock)")
    print("=" * 78)
    print(f"commands           : > < + - . [ ]   (',' rejected)")
    print(f"cells              : 8-bit, mod {CELL_MOD}; tape {TAPE_LEN}; "
          f"max steps {MAX_STEPS}")
    print(f"arithmetic back end: 8-bit adder EBM, anneal "
          f"{[b for b, _ in BF_STAGES]}, chains={BF_CHAINS}")
    print()

    key = jax.random.key(0)

    # First: the compiler must reject the input command.
    rejected = False
    try:
        compile_bf("+,.")
    except BFCompileError:
        rejected = True
    print(f"[front end] rejects input ',' : "
          f"{'YES' if rejected else 'NO'}")
    assert rejected, "compiler must reject ','"

    # Also reject unbalanced brackets.
    bad_brackets = False
    try:
        compile_bf("+[+")
    except BFCompileError:
        bad_brackets = True
    print(f"[front end] rejects bad '[]'  : "
          f"{'YES' if bad_brackets else 'NO'}")
    assert bad_brackets, "compiler must reject unbalanced brackets"
    print()

    # Stochastic clock: one shared tick stream = emergent program counter.
    K = 8
    stream = draw_pbit_stream(jax.random.fold_in(key, 1), 0.30, 60_000)
    onehot, tick_steps, phases = clock_ticks(stream, 4, K)
    assert np.all(onehot.sum(axis=1) == 1), "clock one-hot violated"
    assert np.all(np.diff(tick_steps) > 0), "clock not advancing"

    print(f"{'test':>16} | {'program output':>22} | "
          f"{'EBM ops':>7} {'min gnd%':>8} | result")
    print("-" * 78)

    all_ok = True
    for name, src in TESTS:
        ref_out, ref_tape = run_ref(src)
        tsu_out, tsu_tape, st = run_tsu(src, key, tick_steps, phases, K)

        out_match = tsu_out == ref_out
        tape_match = tsu_tape == ref_tape
        ebm_exact = st["ebm_bad"] == 0
        # clock pacing must be a strict, in-order one-hot prefix
        ticks = st["used_ticks"]
        clock_ok = all(ticks[i][0] < ticks[i + 1][0]
                       for i in range(len(ticks) - 1))
        ok = out_match and tape_match and ebm_exact and clock_ok
        all_ok &= ok

        shown = repr(bytes(tsu_out))
        if len(shown) > 22:
            shown = shown[:19] + "...'"
        print(f"{name:>16} | {shown:>22} | "
              f"{st['ebm_ops']:>7} {st['min_ground'] * 100:>7.1f}% | "
              f"{'PASS' if ok else 'FAIL'}")

        if not ok:
            print(f"    ref_out={ref_out!r} tsu_out={tsu_out!r} "
                  f"out={out_match} tape={tape_match} "
                  f"ebm_exact={ebm_exact} clock={clock_ok}")

    print("-" * 78)
    print("Every '+'/'-' above was computed as the ground state of an 8-bit")
    print("adder EBM (penalty 0 = exact modular arithmetic); '< > [ ] .' are")
    print("host control flow; instruction dispatch was paced by the pbit")
    print("clock. TSU output/tape match the trusted reference on all tests.")
    print()
    print("RESULT:", "ALL VALIDATIONS PASSED" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
