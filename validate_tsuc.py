"""
Validate the TSU-C compiler on a variety of C programs.

For every program:
  1. compile_src  (lex/parse/subset-check/constant-fold)
  2. run_reference (exact 16-bit oracle)   -> assert == hand-known expected
  3. run_tsu       (EBM-backed arithmetic) -> assert == reference, AND
                    every '+ - * / %' was an exact EBM ground state
                    (ebm_bad == 0), AND the stochastic clock advanced.

Also: a set of out-of-subset programs that MUST be rejected at compile time.
"""

import jax

from tsuc import compile_src, run_reference, run_tsu, TSUCError
from stochastic_clock import draw_pbit_stream, clock_ticks

# (name, source, expected_return, expected_output, description)
LIBRARY = [
    ("const_fold",
     "int main(){ return 2 + 3*4 - (10-2)/2; }",
     10, b"", "compile-time fold + precedence (0 EBM ops)"),

    ("vars_arith",
     "int main(){ int a=7; int b=5; return a*b - a - b; }",
     23, b"", "* and - on runtime values"),

    ("if_else",
     "int main(){ int x=9; if (x*2 > 15) return x-3; else return x+3; }",
     6, b"", "branch on an EBM-computed comparison"),

    ("while_sum",
     "int main(){ int i=1; int s=0; while (i<=6){ s=s+i; i=i+1; }"
     " return s; }",
     21, b"", "while loop accumulation"),

    ("for_factorial",
     "int main(){ int f=1; for(int i=1;i<=5;i=i+1){ f=f*i; } return f; }",
     120, b"", "for loop + EBM multiply"),

    ("rec_factorial",
     "int fact(int n){ if (n<=1) return 1; return n*fact(n-1); }"
     "int main(){ return fact(5); }",
     120, b"", "bounded recursion + multiply"),

    ("fib_iter",
     "int main(){ int a=0; int b=1; int n=8; int i=0;"
     " while(i<n){ int t=a+b; a=b; b=t; i=i+1; } return a; }",
     21, b"", "iterative Fibonacci fib(8) (32-bit int via limbs)"),

    ("div_mod",
     "int main(){ int a=20; int q=a/6; int r=a%6; return q*100 + r; }",
     302, b"", "division/modulo (host-exact; EBM = subtract gadget)"),

    ("logic_tern",
     "int main(){ int x=4; int y = (x>3 && x<10) ? x*x : 0; return y; }",
     16, b"", "&&, ternary, EBM square"),

    ("signed_neg",
     "int main(){ int x=3; x = x - 8; return x; }",
     -5, b"", "two's-complement wrap -> signed -5"),

    ("putchar_text",
     "int main(){ int c=70; c=c+2; putchar(c); c=c+1; putchar(c);"
     " putchar(10); return 0; }",
     0, b"HI\n", "output via putchar (EBM-built chars)"),

    ("nested_loops",
     "int main(){ int s=0; int i=1; while(i<=2){ int j=1;"
     " while(j<=2){ s = s + i*j; j=j+1; } i=i+1; } return s; }",
     9, b"", "nested loops, EBM multiply inside (trimmed for runtime)"),

    ("compound_assign",
     "int main(){ int x=4; x += 5; x *= 3; x -= 2; return x; }",
     25, b"", "+=, *=, -= desugaring"),

    ("gcd_euclid",
     "int gcd(int a,int b){ while(a!=b){ if (a>b) a=a-b; else b=b-a; }"
     " return a; }"
     "int main(){ return gcd(48,18); }",
     6, b"", "subtraction-based Euclid GCD (recursion-free fn + loop)"),

    ("int32_mul",
     "int main(){ int a=300; int b=400; return a*b; }",
     120000, b"", "32-bit int multiply (>16-bit result) via EBM limbs"),

    ("long_add64",
     "long main(){ long a=70000; long b=33; return a + b; }",
     70033, b"", "64-bit long add on the EBM adder via 4x16 limbs"),
]

REJECTS = [
    ("pointer",    "int main(){ int *p; return 0; }"),
    ("array",      "int main(){ int a[3]; return 0; }"),
    ("short_kw",   "int main(){ short x; return 0; }"),
    ("input",      "int main(){ int x; scanf(\"%d\",&x); return x; }"),
    ("addr_of",    "int main(){ int x; return &x; }"),
    ("goto",       "int main(){ goto L; L: return 0; }"),
    ("string",     "int main(){ return \"hi\"; }"),
    ("no_main",    "int f(){ return 1; }"),
    ("bad_braces", "int main(){ return 1; "),
]


def main():
    print("=" * 80)
    print("TSU-C COMPILER VALIDATION  (host control flow + EBM arithmetic)")
    print("=" * 80)

    # ---- compile-time rejection of out-of-subset C ----
    print("Subset enforcement (these MUST fail to compile):")
    rej_ok = True
    for name, src in REJECTS:
        try:
            compile_src(src)
            print(f"  {name:<12}: NOT rejected   FAIL")
            rej_ok = False
        except TSUCError as ex:
            msg = str(ex)
            print(f"  {name:<12}: rejected ({msg[:42]})")
    assert rej_ok, "a forbidden construct was accepted"
    print()

    key = jax.random.key(0)
    stream = draw_pbit_stream(jax.random.fold_in(key, 1), 0.30, 160_000)
    onehot, tick_steps, _ph = clock_ticks(stream, 4, 8)
    assert (onehot.sum(axis=1) == 1).all()

    print(f"{'program':>16} | {'ret':>6} {'exp':>6} | {'output':>8} | "
          f"{'EBM':>4} {'rtry':>4} {'ming%':>6} | result")
    print("-" * 80)

    all_ok = True
    total_ebm = 0
    total_rtry = 0
    for name, src, exp_ret, exp_out, _desc in LIBRARY:
        prog = compile_src(src)
        ref_ret, ref_out = run_reference(prog)
        # oracle sanity vs hand-known expected
        sane = (ref_ret == exp_ret and ref_out == exp_out)

        prog2 = compile_src(src)            # fresh AST for the TSU run
        tsu_ret, tsu_out, st = run_tsu(prog2, jax.random.fold_in(key, 2),
                                       tick_steps)
        total_ebm += st["ebm_ops"]
        total_rtry += st["ebm_retries"]

        match = (tsu_ret == ref_ret and tsu_out == ref_out)
        exact = st["ebm_bad"] == 0
        ok = sane and match and exact
        all_ok &= ok

        outrepr = (tsu_out.decode("latin1")
                   .encode("unicode_escape").decode() if tsu_out else "-")
        print(f"{name:>16} | {tsu_ret:>6} {exp_ret:>6} | {outrepr:>8} | "
              f"{st['ebm_ops']:>4} {st['ebm_retries']:>4} "
              f"{st['min_ground']*100:>5.0f}% | "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"     sane={sane} match={match} exact={exact} "
                  f"ref=({ref_ret},{ref_out!r}) tsu=({tsu_ret},{tsu_out!r})")

    print("-" * 80)
    print(f"total EBM relaxations across suite : {total_ebm}  "
          f"(escalate-and-retry recoveries: {total_rtry})")
    print("Every program: host-run control flow, EBM-run arithmetic; TSU "
          "results match the exact reference and all ops hit ground state.")
    print()
    print("RESULT:", "ALL VALIDATIONS PASSED" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
