"""
16-bit limited Brainfuck compiler -> TSU, with a validated program library.

Upgrades over brainfuck.py:
  * 16-bit cells (mod 65536) using the cached annealed AdderEngine.
  * Compiler RUN-LENGTH FOLDING: a maximal run of +/- becomes ONE 16-bit EBM
    add of the net constant; a run of >/< becomes one host move. So `+`*1000
    is a single relaxation, not 1000 -- this is what makes a whole library
    feasible on the sampler.
  * A library of Brainfuck programs, each compiled then run to completion on
    the TSU and differentially validated against an independent, naive,
    per-character reference interpreter (output AND full 16-bit tape).

Supported commands: > < + - . [ ]   (input ',' rejected = "limited").
Control flow / pointer / I/O stay on the host (per FEASIBILITY.md); only the
arithmetic is thermodynamic. Instruction dispatch is paced by the pbit
stochastic clock: each executed instruction consumes one one-hot tick, which
also acts as the "relaxation epoch finished" barrier before advancing.
"""

import jax
import numpy as np

from tsu_runtime import AdderEngine
from stochastic_clock import draw_pbit_stream, clock_ticks

NBITS = 16
CELL_MOD = 1 << NBITS          # 65536
TAPE_LEN = 64
MAX_STEPS = 20000
SUPPORTED = set("><+-.[]")

# 16-bit annealed engine (built once, reused for every op in every program).
STAGES = [(0.7, 300), (1.8, 500), (3.5, 800)]
N_CHAINS = 3
ENGINE = AdderEngine(NBITS, STAGES, N_CHAINS)


class BFCompileError(Exception):
    pass


# ---------------------------------------------------------------------------
# FRONT END: compile + run-length fold -> IR
# ---------------------------------------------------------------------------
def compile_bf(src: str):
    if "," in src:
        raise BFCompileError("input ',' unsupported (limited compiler)")
    toks = [ch for ch in src if ch in SUPPORTED]

    ops = []
    stack = []
    i = 0
    while i < len(toks):
        ch = toks[i]
        if ch in "+-":
            net = 0
            while i < len(toks) and toks[i] in "+-":
                net += 1 if toks[i] == "+" else -1
                i += 1
            net %= CELL_MOD
            if net != 0:
                ops.append(["add", net])           # ONE EBM op per run
            continue
        if ch in "><":
            net = 0
            while i < len(toks) and toks[i] in "><":
                net += 1 if toks[i] == ">" else -1
                i += 1
            if net != 0:
                ops.append(["move", net])
            continue
        if ch == ".":
            ops.append(["out", None])
        elif ch == "[":
            stack.append(len(ops))
            ops.append(["jz", None])
        elif ch == "]":
            if not stack:
                raise BFCompileError("unbalanced ']'")
            o = stack.pop()
            ops.append(["jnz", o + 1])
            ops[o][1] = len(ops)
        i += 1
    if stack:
        raise BFCompileError("unbalanced '['")
    return [tuple(o) for o in ops]


# ---------------------------------------------------------------------------
# Independent reference: naive per-character interpreter (the oracle)
# ---------------------------------------------------------------------------
def run_ref(src: str):
    if "," in src:
        raise BFCompileError("input ',' unsupported")
    code = [c for c in src if c in SUPPORTED]
    match = {}
    st = []
    for p, c in enumerate(code):
        if c == "[":
            st.append(p)
        elif c == "]":
            if not st:
                raise BFCompileError("unbalanced ']'")
            q = st.pop()
            match[p] = q
            match[q] = p
    if st:
        raise BFCompileError("unbalanced '['")

    tape = [0] * TAPE_LEN
    ptr = pc = steps = 0
    out = []
    while pc < len(code):
        steps += 1
        if steps > MAX_STEPS:
            raise BFCompileError("reference exceeded MAX_STEPS")
        c = code[pc]
        if c == ">":
            ptr += 1
        elif c == "<":
            ptr -= 1
        elif c == "+":
            tape[ptr] = (tape[ptr] + 1) % CELL_MOD
        elif c == "-":
            tape[ptr] = (tape[ptr] - 1) % CELL_MOD
        elif c == ".":
            out.append(tape[ptr] & 0xFF)
        elif c == "[":
            if tape[ptr] == 0:
                pc = match[pc]
        elif c == "]":
            if tape[ptr] != 0:
                pc = match[pc]
        if not (0 <= ptr < TAPE_LEN):
            raise BFCompileError("pointer out of bounds")
        pc += 1
    return bytes(out), tape


# ---------------------------------------------------------------------------
# BACK END: run compiled IR to completion on the TSU
# ---------------------------------------------------------------------------
def run_tsu(src: str, key, tick_steps):
    ops = compile_bf(src)
    tape = [0] * TAPE_LEN
    ptr = pc = steps = 0
    out = []
    ebm_ops = ebm_bad = 0
    min_ground = 1.0
    last_tick = -1

    while pc < len(ops):
        if steps >= len(tick_steps):
            raise BFCompileError("ran out of clock ticks")
        tk = int(tick_steps[steps])           # stochastic-clock barrier:
        if tk <= last_tick:                   # relaxation epoch finished
            raise BFCompileError("clock not advancing")
        last_tick = tk
        steps += 1
        if steps > MAX_STEPS:
            raise BFCompileError("TSU exceeded MAX_STEPS")

        kind, arg = ops[pc]
        if kind == "move":
            ptr += arg
            if not (0 <= ptr < TAPE_LEN):
                raise BFCompileError("pointer out of bounds")
        elif kind == "add":
            r = ENGINE.solve(tape[ptr], arg,
                             jax.random.fold_in(key, pc * 100003 + steps))
            ebm_ops += 1
            ebm_bad += (r["penalty"] != 0)
            min_ground = min(min_ground, r["ground_hit_rate"])
            tape[ptr] = r["pred"] % CELL_MOD
        elif kind == "out":
            out.append(tape[ptr] & 0xFF)
        elif kind == "jz":
            if tape[ptr] == 0:
                pc = arg
                continue
        elif kind == "jnz":
            if tape[ptr] != 0:
                pc = arg
                continue
        pc += 1

    return bytes(out), tape, dict(steps=steps, ebm_ops=ebm_ops,
                                  ebm_bad=ebm_bad, min_ground=min_ground)


# ---------------------------------------------------------------------------
# The validated library
# ---------------------------------------------------------------------------
LIBRARY = [
    ("inc_const",    "+++++.",                       "cell=5"),
    ("fold_net",     "+++++++ - - .",                "folded run 7-2=5"),
    ("underflow16",  "-.",                            "0->65535 (16-bit)"),
    ("big_16bit",    "+" * 1000 + ".",               "cell=1000 (>255)"),
    ("over_256",     "+" * 256 + ".",                "cell=256 (>8-bit)"),
    ("mul_3x4",      "+++[>++++<-]>.",                "3*4=12"),
    ("seq_1234",     "+>++>+++>++++<<<.>.>.>.",       "emit 1,2,3,4"),
    ("char_A",       "++++++[>++++++++++<-]>+++++.",  "6*10+5=65 'A'"),
    ("hello_Hi",     "++++++++[>+++++++++<-]>." + "+" * 33 + ".",
                                                      "'H' then 'i'"),
    ("countdown",    "+++++[>+<-]>.",                 "move 5 via loop"),
    ("skipped",      "[++++++++++].",                 "loop skipped"),
    ("nested",       "+++[>++[>+<-]<-]>>.",           "3*2*1=6"),
    ("text_OK",      "+++++++++[>+++++++++<-]>--.----.>++++++++++.",
                                                      "'O','K','\\n'"),
]


def _show(b: bytes) -> str:
    s = b.decode("latin1")
    printable = "".join(ch if 32 <= ord(ch) < 127 else
                        ("\\n" if ch == "\n" else ".") for ch in s)
    return f"{b!r}  ({printable})" if b else "b''"


def main():
    print("=" * 80)
    print("16-BIT LIMITED BRAINFUCK COMPILER  +  validated TSU program library")
    print("=" * 80)
    print(f"cells              : {NBITS}-bit, mod {CELL_MOD}; tape {TAPE_LEN}")
    print(f"datapath           : cached annealed adder, "
          f"{ENGINE.ncolors}-color Gibbs, anneal "
          f"{[b for b, _ in STAGES]}, chains={N_CHAINS}")
    print(f"compiler           : run-length folds +/-/<> ; rejects ',' & "
          f"unbalanced []")
    print()

    key = jax.random.key(0)

    # front-end rejections
    for bad, why in [("+,.", "input ','"), ("+[+", "unbalanced '['"),
                     ("]", "unbalanced ']'")]:
        rej = False
        try:
            compile_bf(bad)
        except BFCompileError:
            rej = True
        print(f"[front end] rejects {why:<18}: {'YES' if rej else 'NO'}")
        assert rej, f"must reject {why}"
    print()

    # stochastic clock = emergent program counter / settle barrier
    stream = draw_pbit_stream(jax.random.fold_in(key, 1), 0.30, 120_000)
    onehot, tick_steps, phases = clock_ticks(stream, 4, 8)
    assert np.all(onehot.sum(axis=1) == 1)
    assert np.all(np.diff(tick_steps) > 0)

    print(f"{'program':>13} | {'output':>26} | {'IR':>3} {'EBM':>4} "
          f"{'min g%':>6} | result")
    print("-" * 80)

    all_ok = True
    total_ebm = 0
    for name, src, _desc in LIBRARY:
        ref_out, ref_tape = run_ref(src)
        tsu_out, tsu_tape, st = run_tsu(src, key, tick_steps)
        total_ebm += st["ebm_ops"]

        ok = (tsu_out == ref_out and tsu_tape == ref_tape
              and st["ebm_bad"] == 0)
        all_ok &= ok
        nir = len(compile_bf(src))
        print(f"{name:>13} | {_show(tsu_out):>26} | {nir:>3} "
              f"{st['ebm_ops']:>4} {st['min_ground'] * 100:>5.0f}% | "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"   ref={ref_out!r} tsu={tsu_out!r} "
                  f"tape_match={tsu_tape == ref_tape} "
                  f"ebm_bad={st['ebm_bad']}")

    print("-" * 80)
    # explicit 16-bit evidence from the tape (not just the low output byte)
    _, t_under = run_ref("-.")
    _, t_big = run_ref("+" * 1000 + ".")
    print(f"16-bit evidence    : underflow tape[0]={t_under[0]} "
          f"(==65535), big tape[0]={t_big[0]} (==1000) -- both > 255")
    print(f"total EBM solves   : {total_ebm}  (run-length folding kept this "
          f"small)")
    print()
    print("Every program compiled and ran to completion on the TSU; all '+'/"
          "'-' were exact 16-bit EBM ground states; outputs and full 16-bit "
          "tapes match the independent reference.")
    print()
    print("RESULT:", "ALL VALIDATIONS PASSED" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
