"""
Hardening / robustness suite for the TSU-C compiler.

Runs entirely on the front end + reference interpreter (NO EBM), so it is
instant and can exhaustively probe robustness:

  A. REJECT  : malformed / out-of-subset programs must fail GRACEFULLY with
               a TSUCError (never a raw Python traceback, never a hang).
  B. CORRECT : tricky-semantics programs must produce exact expected results
               (lexical scoping incl. shadowing, overflow wrap, signed mod,
               short-circuit non-evaluation, do/while, nested break/continue,
               mutual recursion, hex/char/suffix literals, compound bitwise
               assignment, globals, ternary).
  C. LIMITS  : non-terminating / pathological programs must be stopped with
               a TSUCError (step limit, recursion limit, deep-nesting).
  D. REGRESS : the original validated library still has correct reference
               semantics after the scoping refactor.
"""

from tsuc import compile_src, run_reference, TSUCError

REJECT = [
    ("unclosed_brace",     "int main(){ return 1"),
    ("undeclared_use",     "int main(){ int x; return y; }"),
    ("assign_undeclared",  "int main(){ x = 1; return 0; }"),
    ("unknown_call",       "int main(){ return foo(); }"),
    ("arity_mismatch",     "int f(int a){return a;} int main(){return f(1,2);}"),
    ("main_with_params",   "int main(int n){ return n; }"),
    ("dup_function",       "int f(){return 0;} int f(){return 1;}"
                           " int main(){return 0;}"),
    ("redecl_same_scope",  "int main(){ int x; int x; return 0; }"),
    ("dup_param",          "int g(int a,int a){return a;}"
                           " int main(){return g(1,1);}"),
    ("short_kw",           "int main(){ short x; return 0; }"),
    ("float_mod",          "int main(){ float a=1.5; return (int)(a%2); }"),
    ("float_bitnot",       "int main(){ float a=1.5; return ~a != 0; }"),
    ("cast_to_void",       "int main(){ (void)1; return 0; }"),
    ("array",              "int main(){ int a[2]; return 0; }"),
    ("pointer",            "int main(){ int *p; return 0; }"),
    ("address_of",         "int main(){ int x; return &x; }"),
    ("unterminated_cmt",   "int main(){ /* never ends \n return 0; }"),
    ("bad_char_lit",       "int main(){ int c='ab'; return c; }"),
    ("string_lit",         "int main(){ return \"hi\"; }"),
    ("no_main",            "int f(){ return 1; }"),
    ("break_outside",      "int main(){ break; return 0; }"),
    ("continue_outside",   "int main(){ continue; return 0; }"),
    ("malformed_hex",      "int main(){ int x = 0x; return x; }"),
    ("trailing_op",        "int main(){ return 1 +; }"),
    ("bad_assign_target",  "int main(){ 5 = 3; return 0; }"),
    ("goto_kw",            "int main(){ goto L; return 0; }"),
    ("illegal_char",       "int main(){ return 1; } @"),
    ("empty_source",       "   // just a comment\n"),
    ("decl_no_init_use",   "int main(){ int s; s = s + 1; return s; }"),
]
# note: decl_no_init_use is legal C (s defaults to 0); kept as CORRECT below,
# remove from REJECT:
REJECT = [r for r in REJECT if r[0] != "decl_no_init_use"]

CORRECT = [
    ("shadow_inner_discarded",
     "int main(){ int x=1; { int x=99; x=x+1; } return x; }", 1, b""),
    ("inner_assign_propagates",
     "int main(){ int x=1; { x=5; } return x; }", 5, b""),
    ("for_scope_isolated",
     "int main(){ int s=0; for(int i=0;i<4;i=i+1){ s=s+i; }"
     " int i=99; return s+i; }", 105, b""),  # 0+1+2+3=6, +99
    ("int32_no_16bit_wrap",
     "int main(){ int a=40000; return a*a/a; }", 40000, b""),
    ("int32_overflow",
     "int main(){ int a=2000000000; return a + a; }", -294967296, b""),
    ("int_shift_32",
     "int main(){ int x=1; return x << 20; }", 1048576, b""),
    ("long_mul64",
     "long main(){ long a=1000000; return a*a; }", 1000000000000, b""),
    ("long_literal",
     "long main(){ return 5000000000L; }", 5000000000, b""),
    ("long_avoids_overflow",
     "int main(){ int a=2000000000; long b=a; b=b+b;"
     " return (int)(b/1000000000); }", 4, b""),
    ("long_shift64",
     "long main(){ long x=1; return x << 40; }", 1099511627776, b""),
    ("long_factorial15",
     "long fact(int n){ if(n<=1) return 1; return n*fact(n-1); }"
     "long main(){ return fact(15); }", 1307674368000, b""),
    ("long_double_mix",
     "int main(){ long a=2000000000L; double d=a; d=d/1000000000.0;"
     " return (int)(d*10); }", 20, b""),
    ("signed_mod_neg",
     "int main(){ int a=-7; int b=3; return a % b; }", -1, b""),
    ("trunc_div_neg",
     "int main(){ int a=-7; int b=2; return a / b; }", -3, b""),
    ("shortcircuit_and",
     "int beep(){ putchar(66); return 1; }"
     "int main(){ int r = (0 && beep()); return r; }", 0, b""),
    ("shortcircuit_or",
     "int beep(){ putchar(66); return 1; }"
     "int main(){ int r = (1 || beep()); return r; }", 1, b""),
    ("dowhile_runs_once",
     "int main(){ int i=10; int c=0; do { c=c+1; i=i+1; }"
     " while(i<5); return c; }", 1, b""),
    ("nested_break_continue",
     "int main(){ int s=0; int i=0;"
     " while(i<9){ i=i+1; if(i==3) continue; if(i==5) break; s=s+i; }"
     " return s; }", 7, b""),
    ("mutual_recursion",
     "int is_even(int n){ if(n==0) return 1; return is_odd(n-1); }"
     "int is_odd(int n){ if(n==0) return 0; return is_even(n-1); }"
     "int main(){ return is_even(10); }", 1, b""),
    ("hex_char_suffix",
     "int main(){ int a=0x10; int b='A'; int c=1000L; return a+b+c; }",
     1081, b""),
    ("compound_bitwise",
     "int main(){ int x=0xF0; x &= 0x3C; x |= 1; x ^= 2; x <<= 1;"
     " return x; }", 102, b""),
    ("global_and_ternary",
     "int g = 7; int main(){ int x = g>5 ? (g<10?g:0) : -1; return x; }",
     7, b""),
    ("default_zero_init",
     "int main(){ int s; s = s + 1; return s; }", 1, b""),
    ("char_escape_output",
     "int main(){ putchar('H'); putchar('\\n'); return 0; }", 0, b"H\n"),
    ("deep_expr_ok",
     "int main(){ return ((((((((1+1)+1)+1)+1)+1)+1)+1)+1); }", 9, b""),

    # ---- float (Q16.16 fixed-point) ----
    ("float_cast_trunc",
     "int main(){ float x=3.75; return (int)x; }", 3, b""),
    ("float_add_scaled",
     "int main(){ float a=1.5; float b=2.75; return (int)((a+b)*4); }",
     17, b""),
    ("float_sub_neg",
     "int main(){ float a=1.0; float b=2.5; return (int)((a-b)*2); }",
     -3, b""),
    ("float_mul",
     "int main(){ float a=2.5; float b=4.0; return (int)(a*b); }",
     10, b""),
    ("float_div",
     "int main(){ float a=7.0; float b=2.0; return (int)((a/b)*4); }",
     14, b""),
    ("int_float_promote",
     "int main(){ int n=3; float h=0.5; return (int)((n+h)*2); }",
     7, b""),
    ("float_precision",
     "int main(){ float a=0.1; float b=0.2;"
     " return (a+b > 0.299) && (a+b < 0.301); }", 1, b""),
    ("float_ternary",
     "int main(){ float x=2.0; float y = x>1.5 ? x*x : 0.0;"
     " return (int)y; }", 4, b""),
    ("float_loop_accum",
     "int main(){ float s=0.0; int i=0;"
     " while(i<5){ s = s + 0.5; i=i+1; } return (int)(s*2); }",
     5, b""),
    ("float_pi_scaled",
     "int main(){ float p=3.14159; return (int)(p*100); }", 314, b""),
    ("float_fn_return",
     "float half(int n){ return (float)n / 2.0; }"
     "int main(){ return (int)(half(9)*2); }", 9, b""),
    ("float_global",
     "float g = 1.25; int main(){ return (int)(g*8); }", 10, b""),
    ("float_compound",
     "int main(){ float x=1.0; x += 0.5; x *= 4.0; x -= 2.0;"
     " return (int)x; }", 4, b""),

    # ---- double (Q32.32 fixed-point) ----
    ("double_basic",
     "int main(){ double d=3.25; return (int)(d*4); }", 13, b""),
    ("double_div",
     "int main(){ double a=22.0; double b=7.0;"
     " return (int)(a/b*1000); }", 3142, b""),
    ("double_pow2",
     "int main(){ double d = 1.0/256.0; return (int)(d*256.0); }",
     1, b""),
    ("idf_promote",
     "int main(){ int n=2; float f=1.5; double d=2.25;"
     " double r = n + f + d; return (int)(r*4); }", 23, b""),
    ("double_negmul",
     "int main(){ double a=-1.5; double b=3.0; return (int)(a*b); }",
     -4, b""),
    ("double_castchain",
     "int main(){ double q=(double)7/(double)2; return (int)(q*2); }",
     7, b""),
    ("double_global",
     "double g=1.25; int main(){ return (int)(g*8); }", 10, b""),
    ("double_compound",
     "int main(){ double x=10.0; x/=4.0; x+=0.5; return (int)(x*2); }",
     6, b""),
    ("double_fn",
     "double avg(int a,int b){ return ((double)a+(double)b)/2.0; }"
     "int main(){ return (int)(avg(7,10)*2); }", 17, b""),
    ("double_loop",
     "int main(){ double s=0.0; int i=0;"
     " while(i<7){ s=s+0.125; i=i+1; } return (int)(s*8); }", 7, b""),
    ("double_wide_range",
     "int main(){ double d = 250000.0; d = d + 0.5;"
     " return (int)(d / 1000.0); }", 250, b""),
]

LIMITS = [
    ("infinite_loop",      "int main(){ while(1){} return 0; }"),
    ("infinite_for",       "int main(){ for(;;){} return 0; }"),
    ("runaway_recursion",  "int f(int n){ return f(n+1); }"
                           " int main(){ return f(0); }"),
    ("deep_nesting",       "int main(){ return " + "(" * 4000 + "1"
                           + ")" * 4000 + "; }"),
]


def main():
    print("=" * 74)
    print("TSU-C HARDENING SUITE  (front end + reference oracle, no EBM)")
    print("=" * 74)
    ok = True

    print("\nA. Malformed / out-of-subset  -> must raise TSUCError")
    print("-" * 74)
    for name, src in REJECT:
        try:
            compile_src(src)
            print(f"  {name:<22} NOT REJECTED                       FAIL")
            ok = False
        except TSUCError as ex:
            print(f"  {name:<22} ok  ({str(ex)[:40]})")
        except Exception as ex:                    # any non-graceful crash
            print(f"  {name:<22} CRASHED {type(ex).__name__}        FAIL")
            ok = False

    print("\nB. Tricky semantics  -> must equal expected (reference)")
    print("-" * 74)
    for name, src, exp_ret, exp_out in CORRECT:
        try:
            ret, out = run_reference(compile_src(src))
            good = (ret == exp_ret and out == exp_out)
            ok &= good
            print(f"  {name:<24} ret={ret:<6} exp={exp_ret:<6} "
                  f"out={out!r:<8} {'ok' if good else 'FAIL'}")
        except Exception as ex:
            ok = False
            print(f"  {name:<24} CRASHED {type(ex).__name__}: "
                  f"{str(ex)[:30]}  FAIL")

    print("\nC. Pathological  -> must be stopped with TSUCError")
    print("-" * 74)
    for name, src in LIMITS:
        try:
            run_reference(compile_src(src))
            print(f"  {name:<22} NOT STOPPED                        FAIL")
            ok = False
        except TSUCError as ex:
            print(f"  {name:<22} stopped  ({str(ex)[:38]})")
        except RecursionError:
            print(f"  {name:<22} CRASHED RecursionError             FAIL")
            ok = False
        except Exception as ex:
            print(f"  {name:<22} CRASHED {type(ex).__name__}         FAIL")
            ok = False

    print("\nD. Regression  -> original library still correct (reference)")
    print("-" * 74)
    try:
        from validate_tsuc import LIBRARY
        for name, src, er, eo, _d in LIBRARY:
            ret, out = run_reference(compile_src(src))
            good = (ret == er and out == eo)
            ok &= good
            print(f"  {name:<18} ret={ret:<5} exp={er:<5} "
                  f"{'ok' if good else 'FAIL'}")
    except Exception as ex:
        ok = False
        print(f"  regression harness error: {ex!r}  FAIL")

    print("\n" + "=" * 74)
    print("RESULT:", "ALL HARDENING CHECKS PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
