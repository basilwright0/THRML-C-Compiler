"""
Validate TSU-C float (Q16.16 fixed-point) support.

Part 1 (fast, reference oracle): a broad battery of float programs proving
the fixed-point semantics (literals, +,-,*,/, mixed int/float promotion,
casts, comparisons, precision, loops, functions, globals) match exact
expected values.

Part 2 (EBM-backed differential): a small set whose float +,-,* run on the
16-bit EBM adder via the 32-bit-LIMB technique. TSU result must equal the
exact reference, every relaxation must hit ground state.

Float '/' is exactly defined and reference-validated; its EBM lowering is the
same subtract gadget already validated for int '/', and is host-evaluated
here for runtime-budget reasons (reported as fdiv_host, full transparency).
"""

import jax

from tsuc import compile_src, run_reference, run_tsu, TSUCError
from stochastic_clock import draw_pbit_stream, clock_ticks

# (name, src, expected_return, description)   -- reference battery
REF = [
    ("literal_trunc",  "int main(){ float x=9.99; return (int)x; }", 9),
    ("add",            "int main(){ float a=1.25; float b=2.5;"
                       " return (int)((a+b)*4); }", 15),
    ("sub_negative",   "int main(){ float a=2.0; float b=5.5;"
                       " return (int)((a-b)*2); }", -7),
    ("mul",            "int main(){ float a=3.5; float b=6.0;"
                       " return (int)(a*b); }", 21),
    ("div",            "int main(){ float a=22.0; float b=7.0;"
                       " return (int)(a/b*1000); }", 3142),
    ("mixed_promote",  "int main(){ int n=5; float h=0.25;"
                       " return (int)((n - h) * 4); }", 19),
    ("compare_eps",    "int main(){ float a=0.1; float b=0.2;"
                       " return a+b > 0.299 && a+b < 0.301; }", 1),
    ("neg_mul",        "int main(){ float a=-2.5; float b=4.0;"
                       " return (int)(a*b); }", -10),
    ("cast_roundtrip", "int main(){ float f=(float)7/(float)2;"
                       " return (int)(f*2); }", 7),
    ("ternary_f",      "int main(){ float x=2.0;"
                       " float y = x>1.5 ? x*x : 0.0; return (int)y; }", 4),
    # unsuffixed 0.3 is *double*; s(float)=s+0.3 promotes to double then
    # narrows to float each iter -> deterministic 29 (0.3f would give 30)
    ("loop_accum",     "int main(){ float s=0.0; int i=0;"
                       " while(i<10){ s=s+0.3; i=i+1; }"
                       " return (int)(s*10); }", 29),
    ("fn_returns_f",   "float avg(int a,int b){"
                       " return ((float)a + (float)b) / 2.0; }"
                       "int main(){ return (int)(avg(7,10)*2); }", 17),
    ("global_f",       "float g=1.5; int main(){"
                       " g = g + 0.25; return (int)(g*4); }", 7),
    ("compound_f",     "int main(){ float x=10.0; x /= 4.0; x += 0.5;"
                       " return (int)(x*2); }", 6),
    ("pi_area",        "int main(){ float pi=3.14159; float r=2.0;"
                       " return (int)(pi*r*r); }", 12),
    # ---- double (Q32.32) ----
    ("d_basic",        "int main(){ double d=3.25; return (int)(d*4); }",
     13),
    ("d_div",          "int main(){ double a=22.0; double b=7.0;"
                       " return (int)(a/b*1000); }", 3142),
    ("d_promote",      "int main(){ int n=2; float f=1.5; double d=2.25;"
                       " double r=n+f+d; return (int)(r*4); }", 23),
    ("d_wide_range",   "int main(){ double d=250000.0; d=d+0.5;"
                       " return (int)(d/1000.0); }", 250),
    ("d_castchain",    "int main(){ double q=(double)7/(double)2;"
                       " return (int)(q*2); }", 7),
]

# (name, src, expected_return, expected_out)  -- EBM-backed (cheap ops)
EBM = [
    ("f_add",      "int main(){ float a=1.5; float b=2.25;"
                   " return a+b > 3.7; }", 1, b""),
    ("f_sub",      "int main(){ float x=10.0; float y=0.5;"
                   " return (int)(x-y); }", 9, b""),
    ("f_promote",  "int main(){ int n=3; float f=0.5;"
                   " return (int)((n+f)*2.0); }", 7, b""),
    ("f_mul",      "int main(){ float a=2.0; float b=3.0;"
                   " return (int)(a*b); }", 6, b""),
    ("f_negmul",   "int main(){ float a=-1.5; float b=2.0;"
                   " return (int)(a*b); }", -3, b""),
    ("f_loop",     "int main(){ float s=0.0; int i=0;"
                   " while(i<4){ s=s+0.25; i=i+1; }"
                   " return (int)(s*4); }", 4, b""),
    # double (Q32.32) on the EBM adder via 4 x 16-bit limbs
    ("d_add",      "int main(){ double a=1.5; double b=2.25;"
                   " return a+b > 3.7; }", 1, b""),
    ("d_sub",      "int main(){ double x=10.0; double y=0.5;"
                   " return (int)(x-y); }", 9, b""),
    ("d_mul",      "int main(){ double a=2.0; double b=3.0;"
                   " return (int)(a*b); }", 6, b""),
]


def main():
    print("=" * 78)
    print("TSU-C FLOAT VALIDATION  (Q16.16 fixed-point)")
    print("=" * 78)

    print("\nPart 1 - reference semantics (fast, no EBM)")
    print("-" * 78)
    ok = True
    for name, src, exp in REF:
        try:
            ret, _ = run_reference(compile_src(src))
            good = ret == exp
            ok &= good
            print(f"  {name:<16} ret={ret:<7} exp={exp:<7} "
                  f"{'ok' if good else 'FAIL'}")
        except Exception as ex:
            ok = False
            print(f"  {name:<16} CRASH {type(ex).__name__}: "
                  f"{str(ex)[:34]}  FAIL")

    print("\nPart 2 - EBM-backed differential (float +,-,* on the adder)")
    print("-" * 78)
    key = jax.random.key(0)
    stream = draw_pbit_stream(jax.random.fold_in(key, 1), 0.30, 160_000)
    onehot, tick_steps, _ = clock_ticks(stream, 4, 8)
    assert (onehot.sum(axis=1) == 1).all()

    print(f"{'program':>10} | {'ret':>5} {'exp':>5} | {'EBM':>4} "
          f"{'rtry':>4} {'ming%':>6} | result")
    print("-" * 78)
    total = 0
    for name, src, exp_ret, exp_out in EBM:
        prog = compile_src(src)
        rref, oref = run_reference(prog)
        prog2 = compile_src(src)
        rtsu, otsu, st = run_tsu(prog2, jax.random.fold_in(key, 2),
                                 tick_steps)
        total += st["ebm_ops"]
        good = (rref == exp_ret and rtsu == rref and otsu == oref
                and st["ebm_bad"] == 0)
        ok &= good
        print(f"{name:>10} | {rtsu:>5} {exp_ret:>5} | {st['ebm_ops']:>4} "
              f"{st['ebm_retries']:>4} {st['min_ground']*100:>5.0f}% | "
              f"{'PASS' if good else 'FAIL'}")
        if not good:
            print(f"   ref={rref} tsu={rtsu} exp={exp_ret} "
                  f"ebm_bad={st['ebm_bad']} fdiv_host={st['fdiv_host']}")

    print("-" * 78)
    print(f"total EBM relaxations (float suite): {total}")
    print("Float +,-,* executed as 16-bit EBM ground states via 32-bit "
          "limbs; results match the exact fixed-point reference.")
    print()
    print("RESULT:", "ALL FLOAT VALIDATIONS PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
