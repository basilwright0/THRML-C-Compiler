"""
TSU-C: a hardened, limited C compiler whose arithmetic runs on the TSU.

Numeric domains (all fixed-point; IEEE-754 is the Wall-A research line per
FEASIBILITY.md / PLAN_C_COMPILER.md):

    int    : 32-bit two's complement                      (range ~ +/-2.1e9)
    long   : 64-bit two's complement                      (range ~ +/-9.2e18)
    float  : Q16.16  signed 32-bit  (bits / 2^16)          step ~ 1.5e-5
    double : Q32.32  signed 64-bit  (bits / 2^32)          step ~ 2.3e-10

Promotion lattice  int < long < float < double  (common = higher rank;
not byte-identical to ISO usual-arithmetic-conversions but a documented,
deterministic total order; e.g. long+float -> float, as in C).

EBM execution: every value-producing + - (unary -) * in EVERY domain runs on
the *same* validated 16-bit EBM adder via the 16-bit-LIMB technique
(PLAN_C_COMPILER.md §4): int = 2 limbs, long/double = 4, float = 2, with
carry chained host-side. Comparisons / bitwise / shifts / scaling are host
(the plan's free class). Division (all domains) is exactly defined and
reference-validated but host-evaluated for runtime budget; its EBM lowering
is the subtract gadget already validated for integer subtraction (reported
as div_host). Literals fold to fixed-point at compile time (no EBM).
"""

import sys
from fractions import Fraction

import jax

from tsu_runtime import AdderEngine

sys.setrecursionlimit(20_000)

RECURSION_LIMIT = 600
STEP_LIMIT = 500_000

# numeric-domain table
DBITS = {"int": 32, "long": 64, "float": 32, "double": 64}
DFRAC = {"int": 0, "long": 0, "float": 16, "double": 32}
DMOD = {d: 1 << b for d, b in DBITS.items()}
DMASK = {d: (1 << b) - 1 for d, b in DBITS.items()}
DRANK = {"int": 0, "long": 1, "float": 2, "double": 3}
INT_DOMS = {"int", "long"}

NBITS = 16                       # the physical EBM adder width

STAGES = [(0.7, 300), (1.8, 500), (3.5, 800)]
N_CHAINS = 3
_ENGINE = AdderEngine(NBITS, STAGES, N_CHAINS)

_RECOVERY_STAGES = [(0.4, 500), (1.0, 700), (2.2, 1000), (4.0, 1600)]
_RECOVERY = AdderEngine(NBITS, _RECOVERY_STAGES, 10)
MAX_RETRY = 8


class TSUCError(Exception):
    """Any front-end / semantic / runtime fault, reported gracefully."""


# ===========================================================================
# Numeric helpers (domain-parameterised)
# ===========================================================================
def to_signed(v, dom):
    m = DMOD[dom]
    v &= m - 1
    return v - m if v >= m >> 1 else v


def _tdiv(num, den):
    """Integer division truncated toward zero."""
    if num == 0:
        return 0
    if (num < 0) != (den < 0):
        return -(abs(num) // abs(den))
    return abs(num) // abs(den)


def convert(v, src, dst):
    """Value-preserving numeric conversion (truncates toward zero)."""
    if src == dst:
        return v & DMASK[dst]
    r = to_signed(v, src)
    sf, df = DFRAC[src], DFRAC[dst]
    scaled = r * (1 << df)
    q = scaled if sf == 0 else _tdiv(scaled, 1 << sf)
    return q & DMASK[dst]


def quantize(rat: Fraction, dom):
    """Round a rational literal to the domain's fixed-point grid."""
    return int(round(rat * (1 << DFRAC[dom]))) & DMASK[dom]


def _cmp(sx, sy, op):
    return 1 if {
        "==": sx == sy, "!=": sx != sy, "<": sx < sy,
        ">": sx > sy, "<=": sx <= sy, ">=": sx >= sy,
    }[op] else 0


def numbin(op, x, y, dom, ln=0):
    """Constant-fold / host-evaluate a binary op in a numeric domain."""
    mask, f, bits = DMASK[dom], DFRAC[dom], DBITS[dom]
    sx, sy = to_signed(x, dom), to_signed(y, dom)
    if op in ("==", "!=", "<", ">", "<=", ">="):
        return _cmp(sx, sy, op)
    if op == "+":
        return (sx + sy) & mask
    if op == "-":
        return (sx - sy) & mask
    if op == "*":
        return _tdiv(sx * sy, 1 << f) & mask if f else (sx * sy) & mask
    if op in ("/", "%"):
        if sy == 0:
            raise TSUCError(f"line {ln}: {op} by zero")
        if f == 0:                                 # integer domains
            q = _tdiv(sx, sy)
            return q & mask if op == "/" else (sx - q * sy) & mask
        if op == "%":
            raise TSUCError(f"line {ln}: '%' requires integer operands")
        return _tdiv(sx * (1 << f), sy) & mask
    if dom not in INT_DOMS:
        raise TSUCError(f"line {ln}: '{op}' requires integer operands")
    if op == "&":
        return (x & y) & mask
    if op == "|":
        return (x | y) & mask
    if op == "^":
        return (x ^ y) & mask
    if op == "<<":
        return (x << (y & (bits - 1))) & mask
    if op == ">>":
        return (to_signed(x, dom) >> (y & (bits - 1))) & mask
    raise TSUCError(f"line {ln}: bad operator {op}")


# ===========================================================================
# Lexer
# ===========================================================================
KEYWORDS = {"int", "long", "void", "char", "float", "double", "if", "else",
            "while", "for", "do", "return", "break", "continue"}
BANNED_KW = {"struct", "union", "goto", "switch", "unsigned",
             "signed", "sizeof", "typedef", "enum", "static", "const",
             "scanf", "gets", "fgets", "getchar", "printf", "extern",
             "register", "volatile", "short"}
TYPE_KW = {"int", "long", "float", "double", "void"}

_3OPS = {"<<=", ">>="}
_2OPS = {"==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
         "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="}
_1OPS = set("+-*/%(){}[];,=<>!~&|^?:.")
_ESC = {"n": 10, "t": 9, "r": 13, "0": 0, "\\": 92, "'": 39, '"': 34}
INT32_MAX = (1 << 31) - 1


def lex(src: str):
    i, n, line = 0, len(src), 1
    toks = []
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = i + 2
            while j + 1 < n and not (src[j] == "*" and src[j + 1] == "/"):
                if src[j] == "\n":
                    line += 1
                j += 1
            if src[j:j + 2] != "*/":
                raise TSUCError(f"line {line}: unterminated /* comment */")
            i = j + 2
            continue
        if c == "'":
            if i + 1 >= n:
                raise TSUCError(f"line {line}: unterminated char literal")
            if src[i + 1] == "\\":
                if i + 3 >= n or src[i + 3] != "'":
                    raise TSUCError(
                        f"line {line}: bad/unterminated char literal")
                ch = _ESC.get(src[i + 2])
                if ch is None:
                    raise TSUCError(
                        f"line {line}: unknown escape \\{src[i + 2]}")
                toks.append(("NUM", (ch, "int"), line))
                i += 4
                continue
            if i + 2 >= n or src[i + 2] != "'":
                raise TSUCError(
                    f"line {line}: bad/unterminated char literal")
            toks.append(("NUM", (ord(src[i + 1]), "int"), line))
            i += 3
            continue
        if c == '"':
            raise TSUCError(
                f"line {line}: string literals unsupported in TSU-C")
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            if c == "0" and i + 1 < n and src[i + 1] in "xX":
                j = i + 2
                if j >= n or src[j] not in "0123456789abcdefABCDEF":
                    raise TSUCError(f"line {line}: malformed hex literal")
                while j < n and src[j] in "0123456789abcdefABCDEF":
                    j += 1
                val = int(src[i:j], 16)
                is_long = False
                while j < n and src[j] in "uUlL":
                    if src[j] in "lL":
                        is_long = True
                    j += 1
                base = "long" if (is_long or val > 0xFFFFFFFF) else "int"
                toks.append(("NUM", (val, base), line))
                i = j
                continue
            is_float = (c == ".")
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == ".":
                is_float = True
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
            mant = src[i:j]
            exp = 0
            if j < n and src[j] in "eE":
                is_float = True
                j += 1
                k0 = j
                if j < n and src[j] in "+-":
                    j += 1
                if j >= n or not src[j].isdigit():
                    raise TSUCError(
                        f"line {line}: malformed float exponent")
                while j < n and src[j].isdigit():
                    j += 1
                exp = int(src[k0:j])
            if is_float:
                base = "double"
                while j < n and src[j] in "fFlL":
                    if src[j] in "fF":
                        base = "float"
                    j += 1
                m = mant
                if m.startswith("."):
                    m = "0" + m
                if m.endswith("."):
                    m = m + "0"
                rat = Fraction(m) * (Fraction(10) ** exp)
                toks.append(("FNUM", (rat, base), line))
            else:
                val = int(mant)
                is_long = False
                while j < n and src[j] in "uUlL":
                    if src[j] in "lL":
                        is_long = True
                    j += 1
                base = "long" if (is_long or val > INT32_MAX) else "int"
                toks.append(("NUM", (val, base), line))
            if j < n and (src[j].isalpha() or src[j] == "_"):
                raise TSUCError(f"line {line}: malformed numeric literal")
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            w = src[i:j]
            if w in BANNED_KW:
                raise TSUCError(
                    f"line {line}: unsupported C construct '{w}'")
            toks.append(("KW", w, line) if w in KEYWORDS
                        else ("ID", w, line))
            i = j
            continue
        if src[i:i + 3] in _3OPS:
            toks.append(("OP", src[i:i + 3], line))
            i += 3
            continue
        if src[i:i + 2] in _2OPS:
            toks.append(("OP", src[i:i + 2], line))
            i += 2
            continue
        if c in _1OPS:
            toks.append(("OP", c, line))
            i += 1
            continue
        raise TSUCError(f"line {line}: illegal character {c!r}")
    toks.append(("EOF", None, line))
    return toks


# ===========================================================================
# AST + parser
# ===========================================================================
class Node:
    __slots__ = ("kind", "__dict__")

    def __init__(self, kind, line=0, **kw):
        self.kind = kind
        self.line = line
        self.ty = None
        self.__dict__.update(kw)


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.p = 0

    def cur(self):
        return self.t[self.p]

    def line(self):
        return self.t[self.p][2]

    def at(self, kind, val=None):
        k, v, _ = self.t[self.p]
        return k == kind and (val is None or v == val)

    def at_type(self):
        k, v, _ = self.t[self.p]
        return k == "KW" and v in TYPE_KW

    def eat(self, kind, val=None):
        k, v, ln = self.t[self.p]
        if k != kind or (val is not None and v != val):
            want = f"{kind} {val!r}" if val is not None else kind
            raise TSUCError(f"line {ln}: expected {want}, got {k} {v!r}")
        self.p += 1
        return v

    def type_name(self):
        """Parse a type, allowing 'long' / 'long int'."""
        t = self.eat("KW")
        if t == "long":
            if self.at("KW", "int"):
                self.eat("KW", "int")
            return "long"
        return t

    def parse(self):
        funcs, globs = [], []
        while not self.at("EOF"):
            ln = self.line()
            if not self.at_type():
                raise TSUCError(
                    f"line {ln}: expected a declaration, got "
                    f"{self.cur()[0]} {self.cur()[1]!r}")
            ty = self.type_name()
            if self.at("OP", "*"):
                raise TSUCError(f"line {ln}: pointers unsupported in TSU-C")
            name = self.eat("ID")
            if self.at("OP", "("):
                funcs.append(self.func(ty, name, ln))
            else:
                if ty == "void":
                    raise TSUCError(f"line {ln}: 'void' variable invalid")
                while True:
                    if self.at("OP", "["):
                        raise TSUCError(
                            f"line {ln}: arrays unsupported in TSU-C")
                    init = None
                    if self.at("OP", "="):
                        self.eat("OP", "=")
                        init = self.expr()
                    globs.append(Node("global", ln, name=name, init=init,
                                       gty=ty))
                    if self.at("OP", ","):
                        self.eat("OP", ",")
                        name = self.eat("ID")
                        continue
                    break
                self.eat("OP", ";")
        return Node("program", 0, funcs=funcs, globals=globs)

    def func(self, ty, name, ln):
        self.eat("OP", "(")
        params = []
        if not self.at("OP", ")"):
            if self.at("KW", "void") and self.t[self.p + 1][:2] == ("OP", ")"):
                self.eat("KW", "void")
            else:
                while True:
                    if not self.at_type():
                        raise TSUCError(
                            f"line {ln}: expected parameter type")
                    pty = self.type_name()
                    if pty == "void":
                        raise TSUCError(
                            f"line {ln}: 'void' parameter invalid")
                    if self.at("OP", "*"):
                        raise TSUCError(
                            f"line {ln}: pointer params unsupported")
                    params.append((self.eat("ID"), pty))
                    if self.at("OP", ","):
                        self.eat("OP", ",")
                        continue
                    break
        self.eat("OP", ")")
        if self.at("OP", ";"):
            raise TSUCError(
                f"line {ln}: forward declarations unsupported")
        return Node("func", ln, name=name, params=params,
                    body=self.block(), ret=ty)

    def block(self):
        ln = self.line()
        self.eat("OP", "{")
        stmts = []
        while not self.at("OP", "}"):
            if self.at("EOF"):
                raise TSUCError(f"line {ln}: unclosed '{{' block")
            stmts.append(self.stmt())
        self.eat("OP", "}")
        return Node("block", ln, stmts=stmts)

    def stmt(self):
        ln = self.line()
        if self.at("OP", "{"):
            return self.block()
        if self.at("OP", ";"):
            self.eat("OP", ";")
            return Node("empty", ln)
        if self.at("KW", "char"):
            raise TSUCError(f"line {ln}: char variables unsupported")
        if self.at_type() and not self.at("KW", "void"):
            ty = self.type_name()
            return self.decl_tail(ln, ty)
        if self.at("KW", "void"):
            raise TSUCError(f"line {ln}: 'void' variable invalid")
        if self.at("KW", "if"):
            self.eat("KW", "if")
            self.eat("OP", "(")
            cond = self.expr()
            self.eat("OP", ")")
            then = self.stmt()
            els = None
            if self.at("KW", "else"):
                self.eat("KW", "else")
                els = self.stmt()
            return Node("if", ln, cond=cond, then=then, els=els)
        if self.at("KW", "while"):
            self.eat("KW", "while")
            self.eat("OP", "(")
            cond = self.expr()
            self.eat("OP", ")")
            return Node("while", ln, cond=cond, body=self.stmt())
        if self.at("KW", "do"):
            self.eat("KW", "do")
            body = self.stmt()
            self.eat("KW", "while")
            self.eat("OP", "(")
            cond = self.expr()
            self.eat("OP", ")")
            self.eat("OP", ";")
            return Node("dowhile", ln, cond=cond, body=body)
        if self.at("KW", "for"):
            self.eat("KW", "for")
            self.eat("OP", "(")
            init = None
            if self.at_type() and not self.at("KW", "void"):
                ty = self.type_name()
                init = self.decl_tail(ln, ty, no_semi=True)
            elif not self.at("OP", ";"):
                init = Node("expr", ln, e=self.expr())
            self.eat("OP", ";")
            cond = None if self.at("OP", ";") else self.expr()
            self.eat("OP", ";")
            post = None if self.at("OP", ")") else self.expr()
            self.eat("OP", ")")
            return Node("for", ln, init=init, cond=cond, post=post,
                        body=self.stmt())
        if self.at("KW", "return"):
            self.eat("KW", "return")
            e = None if self.at("OP", ";") else self.expr()
            self.eat("OP", ";")
            return Node("return", ln, e=e)
        if self.at("KW", "break"):
            self.eat("KW", "break")
            self.eat("OP", ";")
            return Node("break", ln)
        if self.at("KW", "continue"):
            self.eat("KW", "continue")
            self.eat("OP", ";")
            return Node("continue", ln)
        if self.at("KW", "else"):
            raise TSUCError(f"line {ln}: 'else' without matching 'if'")
        e = self.expr()
        self.eat("OP", ";")
        return Node("expr", ln, e=e)

    def decl_tail(self, ln, ty, no_semi=False):
        decls = []
        while True:
            if self.at("OP", "*"):
                raise TSUCError(f"line {ln}: pointers unsupported")
            nm = self.eat("ID")
            if self.at("OP", "["):
                raise TSUCError(f"line {ln}: arrays unsupported")
            init = None
            if self.at("OP", "="):
                self.eat("OP", "=")
                init = self.expr()
            decls.append((nm, init))
            if self.at("OP", ","):
                self.eat("OP", ",")
                continue
            break
        if not no_semi:
            self.eat("OP", ";")
        return Node("decl", ln, decls=decls, dty=ty)

    def expr(self):
        return self.assign()

    def assign(self):
        left = self.ternary()
        if self.at("OP") and self.cur()[1] in (
                "=", "+=", "-=", "*=", "/=", "%=",
                "<<=", ">>=", "&=", "|=", "^="):
            ln = self.line()
            op = self.eat("OP")
            if left.kind != "var":
                raise TSUCError(
                    f"line {ln}: assignment target must be a variable")
            rhs = self.assign()
            if op != "=":
                rhs = Node("bin", ln, op=op[:-1],
                           a=Node("var", ln, name=left.name), b=rhs)
            return Node("assign", ln, name=left.name, e=rhs)
        return left

    def ternary(self):
        c = self.lor()
        if self.at("OP", "?"):
            ln = self.line()
            self.eat("OP", "?")
            a = self.expr()
            self.eat("OP", ":")
            b = self.ternary()
            return Node("tern", ln, c=c, a=a, b=b)
        return c

    def _binl(self, sub, ops):
        x = sub()
        while self.at("OP") and self.cur()[1] in ops:
            ln = self.line()
            op = self.eat("OP")
            x = Node("bin", ln, op=op, a=x, b=sub())
        return x

    def lor(self):
        return self._binl(self.land, ("||",))

    def land(self):
        return self._binl(self.bor, ("&&",))

    def bor(self):
        return self._binl(self.bxor, ("|",))

    def bxor(self):
        return self._binl(self.band, ("^",))

    def band(self):
        return self._binl(self.eq, ("&",))

    def eq(self):
        return self._binl(self.rel, ("==", "!="))

    def rel(self):
        return self._binl(self.shift, ("<", ">", "<=", ">="))

    def shift(self):
        return self._binl(self.add, ("<<", ">>"))

    def add(self):
        return self._binl(self.mul, ("+", "-"))

    def mul(self):
        return self._binl(self.unary, ("*", "/", "%"))

    def unary(self):
        if (self.at("OP", "(") and self.t[self.p + 1][0] == "KW"
                and self.t[self.p + 1][1] in TYPE_KW
                and (self.t[self.p + 2][:2] == ("OP", ")")
                     or (self.t[self.p + 1][1] == "long"
                         and self.t[self.p + 2][:2] == ("KW", "int")))):
            ln = self.line()
            self.eat("OP", "(")
            to = self.type_name()
            self.eat("OP", ")")
            if to == "void":
                raise TSUCError(f"line {ln}: cast to void unsupported")
            return Node("cast", ln, to=to, e=self.unary())
        if self.at("OP") and self.cur()[1] in ("+", "-", "!", "~"):
            ln = self.line()
            op = self.eat("OP")
            return Node("un", ln, op=op, e=self.unary())
        if self.at("OP", "&"):
            raise TSUCError(
                f"line {self.line()}: address-of '&' unsupported")
        if self.at("OP", "*"):
            raise TSUCError(
                f"line {self.line()}: pointer deref '*' unsupported")
        return self.primary()

    def primary(self):
        ln = self.line()
        if self.at("OP", "("):
            self.eat("OP", "(")
            e = self.expr()
            self.eat("OP", ")")
            return e
        if self.at("NUM"):
            v, base = self.eat("NUM")
            return Node("num", ln, v=v, base=base)
        if self.at("FNUM"):
            rat, base = self.eat("FNUM")
            return Node("fnum", ln, rat=rat, base=base)
        if self.at("ID"):
            name = self.eat("ID")
            if self.at("OP", "("):
                self.eat("OP", "(")
                args = []
                if not self.at("OP", ")"):
                    while True:
                        args.append(self.expr())
                        if self.at("OP", ","):
                            self.eat("OP", ",")
                            continue
                        break
                self.eat("OP", ")")
                return Node("call", ln, name=name, args=args)
            if self.at("OP", "["):
                raise TSUCError(f"line {ln}: arrays unsupported")
            return Node("var", ln, name=name)
        raise TSUCError(
            f"line {ln}: unexpected token {self.cur()[0]} "
            f"{self.cur()[1]!r}")


# ===========================================================================
# Static analysis + type inference
# ===========================================================================
def analyze(prog):
    funcs = {}
    for f in prog.funcs:
        if f.name in funcs:
            raise TSUCError(f"line {f.line}: duplicate function '{f.name}'")
        pnames = [p[0] for p in f.params]
        if len(set(pnames)) != len(pnames):
            raise TSUCError(
                f"line {f.line}: duplicate parameter in '{f.name}'")
        funcs[f.name] = f
    if "putchar" in funcs:
        raise TSUCError("cannot redefine builtin 'putchar'")
    if "main" not in funcs:
        raise TSUCError("no main() function")
    if funcs["main"].params:
        raise TSUCError(
            f"line {funcs['main'].line}: main() must take no parameters")

    fret = {n: f.ret for n, f in funcs.items()}
    fpar = {n: [p[1] for p in f.params] for n, f in funcs.items()}
    gty = {g.name: g.gty for g in prog.globals}

    def conv(e, to):
        if e.ty == to:
            return e
        if e.ty == "void" or to == "void":
            raise TSUCError(f"line {e.line}: invalid use of void value")
        c = Node("conv", e.line, e=e, to=to)
        c.ty = to
        return c

    def common(a, b):
        return a.ty if DRANK[a.ty] >= DRANK[b.ty] else b.ty

    def texpr(e, sc):
        k = e.kind
        if k == "num":
            e.ty = e.base
            e.v &= DMASK[e.ty]
        elif k == "fnum":
            e.ty = e.base
        elif k == "var":
            for s in reversed(sc):
                if e.name in s:
                    e.ty = s[e.name]
                    break
            else:
                if e.name in gty:
                    e.ty = gty[e.name]
                else:
                    raise TSUCError(
                        f"line {e.line}: use of undeclared variable "
                        f"'{e.name}'")
        elif k == "assign":
            tgt = None
            for s in reversed(sc):
                if e.name in s:
                    tgt = s[e.name]
                    break
            if tgt is None:
                tgt = gty.get(e.name)
            if tgt is None:
                raise TSUCError(
                    f"line {e.line}: assignment to undeclared "
                    f"'{e.name}'")
            texpr(e.e, sc)
            e.e = conv(e.e, tgt)
            e.ty = tgt
        elif k == "cast":
            texpr(e.e, sc)
            e.e = conv(e.e, e.to)
            e.ty = e.to
        elif k == "un":
            texpr(e.e, sc)
            if e.op == "~" and e.e.ty not in INT_DOMS:
                raise TSUCError(
                    f"line {e.line}: '~' requires integer operand")
            e.ty = "int" if e.op == "!" else e.e.ty
        elif k == "bin":
            texpr(e.a, sc)
            texpr(e.b, sc)
            op = e.op
            if op in ("&", "|", "^", "<<", ">>", "%"):
                if e.a.ty not in INT_DOMS or e.b.ty not in INT_DOMS:
                    raise TSUCError(
                        f"line {e.line}: '{op}' requires integer operands")
                ct = common(e.a, e.b)
                if op in ("<<", ">>"):
                    e.ty = e.a.ty
                else:
                    e.a = conv(e.a, ct)
                    e.b = conv(e.b, ct)
                    e.ty = ct
            elif op in ("+", "-", "*", "/"):
                ct = common(e.a, e.b)
                e.a = conv(e.a, ct)
                e.b = conv(e.b, ct)
                e.ty = ct
            elif op in ("==", "!=", "<", ">", "<=", ">="):
                ct = common(e.a, e.b)
                e.a = conv(e.a, ct)
                e.b = conv(e.b, ct)
                e.opnd = ct
                e.ty = "int"
            elif op in ("&&", "||"):
                e.ty = "int"
            else:
                raise TSUCError(f"line {e.line}: bad operator {op}")
        elif k == "tern":
            texpr(e.c, sc)
            texpr(e.a, sc)
            texpr(e.b, sc)
            ct = common(e.a, e.b)
            e.a = conv(e.a, ct)
            e.b = conv(e.b, ct)
            e.ty = ct
        elif k == "call":
            if e.name == "putchar":
                if len(e.args) != 1:
                    raise TSUCError(
                        f"line {e.line}: putchar() takes 1 arg")
                texpr(e.args[0], sc)
                e.args[0] = conv(e.args[0], "int")
                e.ty = "int"
            elif e.name in fret:
                want = fpar[e.name]
                if len(e.args) != len(want):
                    raise TSUCError(
                        f"line {e.line}: '{e.name}' expects "
                        f"{len(want)} args, got {len(e.args)}")
                for idx, a in enumerate(e.args):
                    texpr(a, sc)
                    e.args[idx] = conv(a, want[idx])
                e.ty = fret[e.name]
            else:
                raise TSUCError(
                    f"line {e.line}: call to unknown function "
                    f"'{e.name}'")
        else:
            raise TSUCError(f"line {e.line}: bad expression {k}")
        return e

    def decl_(sc, name, ty, ln):
        if name in sc[-1]:
            raise TSUCError(
                f"line {ln}: redeclaration of '{name}' in same scope")
        sc[-1][name] = ty

    def tstmt(s, sc, ret_ty, in_loop):
        k = s.kind
        if k == "block":
            sc.append({})
            for st in s.stmts:
                tstmt(st, sc, ret_ty, in_loop)
            sc.pop()
        elif k == "decl":
            for i, (nm, init) in enumerate(s.decls):
                if init is not None:
                    texpr(init, sc)
                    s.decls[i] = (nm, conv(init, s.dty))
                decl_(sc, nm, s.dty, s.line)
        elif k == "expr":
            texpr(s.e, sc)
        elif k == "empty":
            pass
        elif k == "if":
            texpr(s.cond, sc)
            tstmt(s.then, sc, ret_ty, in_loop)
            if s.els is not None:
                tstmt(s.els, sc, ret_ty, in_loop)
        elif k in ("while", "dowhile"):
            texpr(s.cond, sc)
            tstmt(s.body, sc, ret_ty, True)
        elif k == "for":
            sc.append({})
            if s.init is not None:
                tstmt(s.init, sc, ret_ty, in_loop)
            if s.cond is not None:
                texpr(s.cond, sc)
            if s.post is not None:
                texpr(s.post, sc)
            tstmt(s.body, sc, ret_ty, True)
            sc.pop()
        elif k == "return":
            if s.e is not None:
                texpr(s.e, sc)
                if ret_ty != "void":
                    s.e = conv(s.e, ret_ty)
        elif k in ("break", "continue"):
            if not in_loop:
                raise TSUCError(
                    f"line {s.line}: '{k}' outside of a loop")
        else:
            raise TSUCError(f"line {s.line}: bad statement {k}")

    for f in prog.funcs:
        tstmt(f.body, [{p[0]: p[1] for p in f.params}], f.ret, False)
    for g in prog.globals:
        if g.init is not None:
            texpr(g.init, [{}])
            g.init = conv(g.init, g.gty)


# ===========================================================================
# Constant folding (type-aware)
# ===========================================================================
def fold(node):
    if not isinstance(node, Node):
        return node
    for k, v in list(node.__dict__.items()):
        if isinstance(v, Node):
            setattr(node, k, fold(v))
        elif isinstance(v, list):
            out = []
            for x in v:
                if isinstance(x, Node):
                    out.append(fold(x))
                elif isinstance(x, tuple):
                    out.append(tuple(fold(z) if isinstance(z, Node) else z
                                     for z in x))
                else:
                    out.append(x)
            setattr(node, k, out)

    if node.kind == "fnum":
        node.kind, node.v = "num", quantize(node.rat, node.ty)
    elif node.kind in ("conv", "cast") and node.e.kind == "num":
        node.kind = "num"
        node.v = convert(node.e.v, node.e.ty, node.to)
        node.ty = node.to
    elif node.kind == "un" and node.e.kind == "num":
        d, x = node.e.ty, node.e.v
        if node.op == "-":
            node.kind, node.v = "num", (-to_signed(x, d)) & DMASK[d]
        elif node.op == "+":
            node.kind, node.v = "num", x
        elif node.op == "!":
            node.kind, node.v = "num", (0 if to_signed(x, d) != 0 else 1)
        elif node.op == "~":
            node.kind, node.v = "num", (~x) & DMASK[d]
    elif (node.kind == "bin" and node.a.kind == "num"
          and node.b.kind == "num" and node.op not in ("&&", "||")):
        dom = node.a.ty if node.op not in ("<<", ">>") else node.a.ty
        node.kind, node.v = "num", numbin(node.op, node.a.v, node.b.v,
                                          dom, node.line)
    return node


def compile_src(src: str) -> Node:
    try:
        prog = Parser(lex(src)).parse()
        analyze(prog)
        for f in prog.funcs:
            fold(f.body)
        for g in prog.globals:
            if g.init is not None:
                g.init = fold(g.init)
                if g.init.kind != "num":
                    raise TSUCError(
                        f"line {g.line}: global initializer must be "
                        f"constant")
        return prog
    except TSUCError:
        raise
    except RecursionError:
        raise TSUCError("program too deeply nested to compile")
    except Exception as ex:                       # pragma: no cover
        raise TSUCError(f"internal compile error: {ex!r}")


# ===========================================================================
# Lexically-scoped environment
# ===========================================================================
class Env:
    __slots__ = ("scopes", "globals")

    def __init__(self, globals_, base):
        self.scopes = [base]
        self.globals = globals_

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def declare(self, name, val):
        self.scopes[-1][name] = val

    def get(self, name, ln):
        for sc in reversed(self.scopes):
            if name in sc:
                return sc[name]
        if name in self.globals:
            return self.globals[name]
        raise TSUCError(f"line {ln}: undefined variable '{name}'")

    def set(self, name, val, ln):
        for sc in reversed(self.scopes):
            if name in sc:
                sc[name] = val
                return
        if name in self.globals:
            self.globals[name] = val
            return
        raise TSUCError(f"line {ln}: assignment to undefined '{name}'")


# ===========================================================================
# Interpreter (host vs EBM arithmetic kernel)
# ===========================================================================
class _Ret(Exception):
    def __init__(self, v):
        self.v = v


class _Brk(Exception):
    pass


class _Cont(Exception):
    pass


class Interp:
    def __init__(self, prog, use_ebm=False, key=None, tick_steps=None):
        self.funcs = {f.name: f for f in prog.funcs}
        self.globals = {}
        for g in prog.globals:
            self.globals[g.name] = (g.init.v if g.init is not None else 0)
        self.use_ebm = use_ebm
        self.key = key
        self.ticks = tick_steps
        self.output = bytearray()
        self.steps = 0
        self.depth = 0
        self.ebm_ops = 0
        self.ebm_bad = 0
        self.ebm_retries = 0
        self.div_host = 0
        self.min_ground = 1.0
        self.last_tick = -1

    def _tick(self):
        if self.ticks is None:
            return
        if self.ebm_ops >= len(self.ticks):
            raise TSUCError("ran out of stochastic-clock ticks")
        tk = int(self.ticks[self.ebm_ops])
        if tk <= self.last_tick:
            raise TSUCError("stochastic clock not advancing")
        self.last_tick = tk

    def _solve(self, a, b):
        """Exact 16-bit EBM add (returns a+b incl. carry); retry to ground."""
        a &= 0xFFFF
        b &= 0xFFFF
        self._tick()
        base = (a * 131071 + b * 977 + self.ebm_ops) & 0x7FFFFFFF
        r = _ENGINE.solve(a, b, jax.random.fold_in(self.key, base))
        self.ebm_ops += 1
        if r["ground_hit_rate"] < self.min_ground:
            self.min_ground = r["ground_hit_rate"]
        attempt = 0
        while r["penalty"] != 0 and attempt < MAX_RETRY:
            self.ebm_retries += 1
            attempt += 1
            eng = _ENGINE if attempt < 3 else _RECOVERY
            r = eng.solve(a, b, jax.random.fold_in(
                self.key, (base ^ (0x9E3779B1 * attempt)) & 0x7FFFFFFF))
        if r["penalty"] != 0:
            self.ebm_bad += 1
            raise TSUCError(f"EBM add {a}+{b} failed to reach ground")
        return r["pred"]

    def _wadd(self, a, b):
        """Exact non-negative wide add via 16-bit EBM limbs (+carry)."""
        if not self.use_ebm:
            return a + b
        res = shift = carry = 0
        while a or b or carry:
            t = self._solve(a & 0xFFFF, b & 0xFFFF)
            limb, cout = t & 0xFFFF, t >> 16
            if carry:
                t2 = self._solve(limb, 1)
                limb, cout = t2 & 0xFFFF, cout + (t2 >> 16)
            res |= limb << shift
            shift += 16
            carry = 1 if cout else 0
            a >>= 16
            b >>= 16
        return res

    # ---- domain-generic arithmetic ----
    def dadd(self, a, b, dom):
        m = DMASK[dom]
        if not self.use_ebm:
            return (a + b) & m
        return self._wadd(a & m, b & m) & m

    def dsub(self, a, b, dom):
        m = DMASK[dom]
        return self.dadd(a & m, (-to_signed(b, dom)) & m, dom)

    def dmul(self, a, b, dom):
        m, f = DMASK[dom], DFRAC[dom]
        sa, sb = to_signed(a, dom), to_signed(b, dom)
        neg = (sa < 0) != (sb < 0)
        ua, ub = abs(sa), abs(sb)
        if not self.use_ebm:
            p = ua * ub
        else:
            p, i = 0, 0
            while ub >> i:
                if (ub >> i) & 1:
                    p = self._wadd(p, ua << i)
                i += 1
        mag = p >> f if f else p
        return (-mag if neg else mag) & m

    def ddiv(self, a, b, dom, ln, want_mod=False):
        sb = to_signed(b, dom)
        if sb == 0:
            raise TSUCError(f"line {ln}: {'%' if want_mod else '/'} "
                            f"by zero")
        self.div_host += 1
        sa = to_signed(a, dom)
        f = DFRAC[dom]
        if want_mod:
            q = _tdiv(sa, sb)
            return (sa - q * sb) & DMASK[dom]
        return _tdiv(sa * (1 << f), sb) & DMASK[dom]

    # ---- evaluation ----
    def _truthy(self, e, env):
        return to_signed(self.eval(e, env), e.ty) != 0

    def eval(self, e, env):
        k = e.kind
        if k == "num":
            return e.v
        if k == "var":
            return env.get(e.name, e.line)
        if k == "assign":
            v = self.eval(e.e, env)
            env.set(e.name, v, e.line)
            return v
        if k == "conv" or k == "cast":
            return convert(self.eval(e.e, env), e.e.ty, e.to)
        if k == "un":
            if e.op == "-":
                return self.dsub(0, self.eval(e.e, env), e.ty)
            x = self.eval(e.e, env)
            if e.op == "+":
                return x
            if e.op == "!":
                return 0 if to_signed(x, e.e.ty) != 0 else 1
            return (~x) & DMASK[e.ty]
        if k == "bin":
            op = e.op
            if op == "&&":
                return 1 if (self._truthy(e.a, env)
                             and self._truthy(e.b, env)) else 0
            if op == "||":
                if self._truthy(e.a, env):
                    return 1
                return 1 if self._truthy(e.b, env) else 0
            a = self.eval(e.a, env)
            b = self.eval(e.b, env)
            if op in ("==", "!=", "<", ">", "<=", ">="):
                return _cmp(to_signed(a, e.opnd),
                            to_signed(b, e.opnd), op)
            if op == "+":
                return self.dadd(a, b, e.ty)
            if op == "-":
                return self.dsub(a, b, e.ty)
            if op == "*":
                return self.dmul(a, b, e.ty)
            if op == "/":
                return self.ddiv(a, b, e.ty, e.line)
            if op == "%":
                return self.ddiv(a, b, e.ty, e.line, want_mod=True)
            return numbin(op, a, b, e.ty, e.line)     # bitwise / shift
        if k == "tern":
            return (self.eval(e.a, env) if self._truthy(e.c, env)
                    else self.eval(e.b, env))
        if k == "call":
            return self.call(e.name, [self.eval(a, env) for a in e.args],
                             e.line)
        raise TSUCError(f"line {e.line}: bad expression {k}")

    def call(self, name, args, ln):
        if name == "putchar":
            self.output.append(args[0] & 0xFF)
            return args[0] & 0xFF
        if name not in self.funcs:
            raise TSUCError(f"line {ln}: undefined function '{name}'")
        f = self.funcs[name]
        if len(args) != len(f.params):
            raise TSUCError(f"line {ln}: '{name}' arg count mismatch")
        self.depth += 1
        if self.depth > RECURSION_LIMIT:
            raise TSUCError("recursion depth limit exceeded")
        env = Env(self.globals,
                  {p[0]: a for p, a in zip(f.params, args)})
        try:
            self.exec(f.body, env)
            ret = 0
        except _Ret as r:
            ret = r.v
        self.depth -= 1
        return ret

    def exec(self, s, env):
        self.steps += 1
        if self.steps > STEP_LIMIT:
            raise TSUCError("step limit exceeded (non-terminating?)")
        k = s.kind
        if k == "block":
            env.push()
            try:
                for st in s.stmts:
                    self.exec(st, env)
            finally:
                env.pop()
            return
        if k == "empty":
            return
        if k == "decl":
            for nm, init in s.decls:
                env.declare(nm, self.eval(init, env) if init is not None
                            else 0)
            return
        if k == "expr":
            self.eval(s.e, env)
            return
        if k == "if":
            if self._truthy(s.cond, env):
                self.exec(s.then, env)
            elif s.els is not None:
                self.exec(s.els, env)
            return
        if k == "while":
            while self._truthy(s.cond, env):
                try:
                    self.exec(s.body, env)
                except _Brk:
                    break
                except _Cont:
                    continue
            return
        if k == "dowhile":
            while True:
                try:
                    self.exec(s.body, env)
                except _Brk:
                    break
                except _Cont:
                    pass
                if not self._truthy(s.cond, env):
                    break
            return
        if k == "for":
            env.push()
            try:
                if s.init is not None:
                    self.exec(s.init, env)
                while s.cond is None or self._truthy(s.cond, env):
                    try:
                        self.exec(s.body, env)
                    except _Brk:
                        break
                    except _Cont:
                        pass
                    if s.post is not None:
                        self.eval(s.post, env)
            finally:
                env.pop()
            return
        if k == "return":
            raise _Ret(self.eval(s.e, env) if s.e is not None else 0)
        if k == "break":
            raise _Brk()
        if k == "continue":
            raise _Cont()
        raise TSUCError(f"line {s.line}: bad statement {k}")

    def run_main(self):
        try:
            ret = self.call("main", [], 0)
        except (_Brk, _Cont):
            raise TSUCError("'break'/'continue' outside of a loop")
        except RecursionError:
            raise TSUCError("recursion too deep (host stack)")
        mt = self.funcs["main"].ret
        return to_signed(ret, mt if mt in DBITS else "int"), \
            bytes(self.output)


def run_reference(prog):
    return Interp(prog, use_ebm=False).run_main()


def run_tsu(prog, key, tick_steps):
    it = Interp(prog, use_ebm=True, key=key, tick_steps=tick_steps)
    ret, out = it.run_main()
    return ret, out, dict(ebm_ops=it.ebm_ops, ebm_bad=it.ebm_bad,
                          ebm_retries=it.ebm_retries,
                          fdiv_host=it.div_host, div_host=it.div_host,
                          min_ground=it.min_ground, steps=it.steps)
