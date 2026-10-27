"""The evidence ledger for Parts IX-XIV, generated from the checks themselves.

Each Part's ``status_*`` function prints one line per check. This module
runs those functions, reads the lines back, and gives every check a *kind*
from a closed vocabulary:

    D  derived      exact arithmetic (integer, rational or symbolic)
    N  numerical    floating point, at a stated tolerance
    C  counted      a finite enumeration (a box, a patch, a grid) that does
                    not cover the whole range of the claim
    X  second route re-derived by code that shares nothing with the first
    Q  quoted       rests on a named result from the literature
    P  premise      carries a modelling step or an open parameter
    K  control      a negative check: something that must fail does
    L  Lean         kernel-checked facts in sixsphere_lean cover its
                    derived part (a quoted input stays Q)

The table (:data:`KIND`) is hand-written, because "what does this check
rest on" is a judgement. What is mechanical is that it is complete. Every
printed check needs a row and every row needs a printed check. Every ``Q``
needs a named source and every ``X`` a named second route, and every ``L``
names facts of :data:`pyCICY.sixsphere_lean.JACOBI_FACTS`. A ``P`` row is
never linked to Lean, and Lean never removes a ``Q``: on a quoted row it
covers only the derived part, and the quote keeps its source. The approach is ported from
``spectre_ledger.py`` in the spectre repository, which also added ``N`` and
``K``: most checks from Part X on are floating point, and the controls
should be visible as controls.

The point is :data:`CRITICAL`, the checks that Theorem XI
(``F(g x) = e(chi(g)/12) F(x)``, ``<T1, T2> = ker chi``) rests on. On that
path :func:`critical_quoted` must list exactly the quoted inputs, and
nothing else. There are two:

- the classical theta_1 and eta laws (Mumford, Tata I);
- that S and T generate SL(2, Z).

Everything else on that path is kernel-checked in Lean 4 (lead 5): the
Heisenberg law, f a homomorphism for all integers, q GL(2, F2)-invariant, the
relations alpha satisfies, chi(T1) = chi(T2) = 0 and chi(z) = 1, the holonomy
words, the generation of {f = 0} for all (x, y, s), and the five generator
exponents of Part XI. What Lean does not see is the group-theory step (v) of
``holonomy_index`` and the two quoted inputs.

    python3 -m pyCICY.sixsphere_ledger            # table
    python3 -m pyCICY.sixsphere_ledger --write    # refresh the block in AlpogeSixSphere.md
"""
import contextlib
import io
import os
import re
from collections import Counter

__all__ = ["SOURCES", "KIND", "QUOTED", "SECOND_ROUTE", "LEAN", "CRITICAL",
           "VOCABULARY", "collect", "selftest", "critical_quoted", "counts",
           "markdown", "write_markdown", "status_ledger"]

VOCABULARY = "DNCXQPKL"

# (part, short name, module, status function)
SOURCES = [
    ("IX", "gauge", "sixsphere_gauge", "status_gauge"),
    ("X", "hofstadter", "cusp_hofstadter", "status_cusp_hofstadter"),
    ("XI", "theta", "sixsphere_theta", "status_theta"),
    ("XII", "classes", "sixsphere_classes", "status_classes"),
    ("XIII", "lll", "cusp_lll", "status_lll"),
    ("XIV", "mirror", "cusp_mirror", "status_mirror"),
]

KIND = {
    # Part IX: the structure group and the holonomy
    ("gauge", "invariant tensors: gamma, delta^, Q0 unique"): "D",
    ("gauge", "delta^ = Q0(gamma, .)"): "D",
    ("gauge", "commutant <I, gamma (x) delta^>"): "D",
    ("gauge", "automorphisms of (Q0, gamma) = Jacobi group, r forced"): "D",
    ("gauge", "<T1, T2> = ker chi, index 12"): "DQL",
    ("gauge", "control: Levi lifts have chi = (4, 9), cusp rank 1"): "DK",
    ("gauge", "Zariski closure: Lie algebra dim 6 = stabiliser"): "D",
    ("gauge", "G(R) keeps the 6 mu shape and preserves D"): "D",
    ("gauge", "orbit dim 5, stabiliser SO(2)"): "D",
    ("gauge", "four classes = d in {1,2,3,6}, Pf 6; d | p, least |p| = d"): "C",
    # Part X: electrons on the cusp fibre
    ("hofstadter", "6 sites (C, 3E, 2V), 12 bonds per domain"): "DX",
    ("hofstadter", "charpoly = E^6 - (12+s)E^4 - 3(s-6)E^2 - |Delta_A2|^2"): "DX",
    ("hofstadter", "det B = unit x Weyl denominator"): "D",
    ("hofstadter", "E = 0 exactly on the three mirror lines"): "DC",
    ("hofstadter", "B B^T row sums 18 at k = 0"): "D",
    ("hofstadter", "spectrum fills [-3 sqrt 2, 3 sqrt 2]"): "N",
    ("hofstadter", "every kite carries flux phi"): "C",
    ("hofstadter", "TKNN integer in every gap at phi = 1/3"): "NX",
    # Part XI: the holonomy character as a theta multiplier
    ("theta", "F(g x) = e(chi(g)/12) F(x) on the 7 generators"): "N",
    ("theta", "... and on 20 random words"): "N",
    ("theta", "alpha by Euclid = alpha from the eta^2 multiplier"): "DN",
    ("theta", "<T1, T2> fixes F; g fixes F iff chi(g) = 0; F^12 invariant"): "D",
    ("theta", "z^k reach all 12 multipliers"): "D",
    ("theta", "80 candidates, survivors exactly theta_1 / eta with e(beta/12)"): "N",
    ("theta", "control: e(beta/6) theta_1 / eta not fixed by T1"): "NK",
    ("theta", "proof: classical laws + beta shift = chi/12 exactly, 5 generators"): "DQL",
    ("theta", "control: without the beta shift, L(S) and h(0,1,0) are not constant"): "DK",
    ("theta", "the quoted classical laws hold against the series"): "N",
    ("theta", "F = 0 on mu in Z + tau Z, nonzero at half periods"): "QN",
    # Part XII: the near-miss classes are isogenies
    ("classes", "normal form d E03 + (6/d) E12, Pf 6, P unimodular, 4 classes"): "D",
    ("classes", "D P^-1 T_i P D^-1 = (W1, W2) for all four classes"): "D",
    ("classes", "h(0,1,0) conjugates (W1, W2) onto the paper's (T1, T2)"): "D",
    ("classes", "Q-conjugate to the paper by C_d, det +-1/d^2"): "D",
    ("classes", "Phi_d: Aut(Q_d, gamma)(Z) = G(Z) (symbolic + box)"): "DC",
    ("classes", "<W1, W2> = ker chi: index 12 for every class"): "D",
    ("classes", "exact: gcd of reachable p = d, least |p| = d"): "D",
    ("classes", "class laws have the Z_d shape; F_d multiplier = chi o Phi_d"): "DN",
    ("classes", "F_d = 0 on d(Z + tau Z); nonzero at 1, tau, 1+tau for d > 1"): "N",
    ("classes", "control: only characteristic (0, 0) survives; naive theta fails"): "NK",
    # Part XIII: the lowest Landau level
    ("lll", "lowest band isolated in [-2 sqrt 3, -sqrt 6], next at -sqrt 3"): "DN",
    ("lll", "Chern 1 here; -1 in Part X's orientation, 1-cell and supercell"): "NX",
    ("lll", "Phi = -1 states unfold onto Part X's supercell; Phi = +1 do not"): "N",
    ("lll", "band ends: top at the C-zero twist, bottom at the V-zero twist"): "N",
    ("lll", "each site amplitude vanishes at k = (1/2 - v, 1/2 + u)"): "N",
    ("lll", "and nowhere else on the twist torus"): "C",
    ("lll", "the C site at the odd twist (1/2, 1/2)"): "N",
    ("lll", "continuum theta_1 LLL: the same twist law"): "N",
    ("lll", "lattice-continuum overlap > 0.97"): "N",
    ("lll", "controls: no field Chern 0; periodic twist zero on E, not C"): "NK",
    # Part XIV: the substrate against the quantised dP6 hexagon
    ("mirror", "Delta Delta-bar = 12 - s^2 + 4u, u the long-root hexagon"): "D",
    ("mirror", "u is the hexagon on the index-3 sublattice; s, u independent"): "DN",
    ("mirror", "H^2 = [[6 + S, [2]_q K], [., 3 + H_V]] on C u V (patch)"): "N",
    ("mirror", "H^2_CC = 6 + S on the magnetic supercell"): "N",
    ("mirror", "S = quantised B_3 hexagon at Phi = 6 phi mod 2 (moments n <= 10)"): "NX",
    ("mirror", "phi = 1/2: C-V decouple, S pi-flux triangular, gapless on [-3, 3]"): "N",
    ("mirror", "integer Phi: [S, U] = 0 and V flux integer, yet gap Chern +-1"): "N",
    ("mirror", "naive Weyl charpoly: exact at zero field, fails in a field"): "NK",
}

_SL2Z = "S and T generate SL(2, Z) (Serre, A Course in Arithmetic, VII.1)"
_THETA = "theta_1 and eta transformation laws (Mumford, Tata Lectures on Theta I, I.5 and I.7)"
QUOTED = {
    ("gauge", "<T1, T2> = ker chi, index 12"): _SL2Z,
    ("theta", "proof: classical laws + beta shift = chi/12 exactly, 5 generators"): _THETA,
    ("theta", "F = 0 on mu in Z + tau Z, nonzero at half periods"):
        "zeros of theta_1 (Mumford, Tata I, I.5)",
}

_QL = "quantum-lattice extras/ql_cusp_substrate.py (pyqula, retyped geometry)"
SECOND_ROUTE = {
    ("hofstadter", "6 sites (C, 3E, 2V), 12 bonds per domain"): _QL,
    ("hofstadter", "charpoly = E^6 - (12+s)E^4 - 3(s-6)E^2 - |Delta_A2|^2"): _QL,
    ("hofstadter", "TKNN integer in every gap at phi = 1/3"): _QL,
    ("lll", "Chern 1 here; -1 in Part X's orientation, 1-cell and supercell"):
        _QL + "; Part X's supercell after an exact unfolding",
    ("mirror", "S = quantised B_3 hexagon at Phi = 6 phi mod 2 (moments n <= 10)"):
        "pyCICY/quantum_curve.py, which builds HSX's operator from the toric polygon alone",
}

# Lean facts in sixsphere_lean.JACOBI_FACTS that cover a check's derived part.
LEAN = {
    ("gauge", "<T1, T2> = ker chi, index 12"):
        ("heis_group_law", "f_homomorphism", "q_gl2f2_invariant",
         "alpha_relations", "chi_values", "holonomy_words", "f_zero_generated"),
    ("theta", "proof: classical laws + beta shift = chi/12 exactly, 5 generators"):
        ("theta_generator_exponents",),
}

# The checks Theorem XI rests on: the multiplier law, and that its kernel is <T1, T2>.
CRITICAL = [
    ("gauge", "<T1, T2> = ker chi, index 12"),
    ("theta", "proof: classical laws + beta shift = chi/12 exactly, 5 generators"),
    ("theta", "<T1, T2> fixes F; g fixes F iff chi(g) = 0; F^12 invariant"),
]

_LINE = re.compile(r"^  (.*?)\s+(ok|FAIL)$")


def collect():
    """Run every status function; return its parsed check lines, in order."""
    import importlib
    rows = []
    for part, short, mod, fn in SOURCES:
        m = importlib.import_module("pyCICY." + mod)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = getattr(m, fn)()
        lines = [_LINE.match(l) for l in buf.getvalue().splitlines()]
        lines = [x for x in lines if x]
        for x in lines:
            rows.append({"part": part, "source": short, "name": x.group(1),
                         "ok": x.group(2) == "ok"})
        rows.append({"part": part, "source": short, "name": None, "ok": bool(result)})
    return rows


def selftest(rows=None):
    """The mechanical assertions. Returns a dict of named booleans."""
    rows = collect() if rows is None else rows
    checks = [(r["source"], r["name"]) for r in rows if r["name"] is not None]
    seen = Counter(checks)
    kinds = KIND.values()
    return {
        "every check passes": all(r["ok"] for r in rows),
        "every printed check has a row": all(c in KIND for c in checks),
        "every row has a printed check": all(k in seen for k in KIND),
        "check names unique": all(v == 1 for v in seen.values()),
        "kinds in the vocabulary": all(k and set(k) <= set(VOCABULARY) for k in kinds),
        "every Q has a source, every source a Q":
            {k for k, v in KIND.items() if "Q" in v} == set(QUOTED),
        "every X has a route, every route an X":
            {k for k, v in KIND.items() if "X" in v} == set(SECOND_ROUTE),
        "L only where a Lean fact is named":
            {k for k, v in KIND.items() if "L" in v} == set(LEAN),
        "no P row linked to Lean":
            not any("P" in KIND[k] for k in LEAN),
        "every Lean link names a JACOBI_FACTS fact": _lean_names_exist(),
        "controls named as controls":
            all(("K" in v) == ("control" in k[1].lower() or "fails" in k[1])
                for k, v in KIND.items()),
        "critical path covered": all(c in KIND for c in CRITICAL),
    }


def _lean_names_exist():
    from .sixsphere_lean import JACOBI_FACTS
    names = {f.name for f in JACOBI_FACTS}
    return all(v and set(v) <= names for v in LEAN.values())


def critical_quoted():
    """The quoted inputs on Theorem XI's critical path."""
    return sorted({QUOTED[c] for c in CRITICAL if "Q" in KIND[c]})


def counts():
    """Rows per kind letter, per Part and in total."""
    out = {}
    for part, short, _, _ in SOURCES:
        c = Counter()
        for (s, _), v in KIND.items():
            if s == short:
                c.update(v)
        out[part] = {k: c[k] for k in VOCABULARY}
    out["total"] = {k: sum(out[p][k] for p, *_ in SOURCES) for k in VOCABULARY}
    out["rows"] = len(KIND)
    return out


_NAMES = {"D": "derived", "N": "numerical", "C": "counted", "X": "second route",
          "Q": "quoted", "P": "premise", "K": "control", "L": "Lean"}
BEGIN = "<!-- sixsphere_ledger: generated, do not edit by hand -->"
END = "<!-- /sixsphere_ledger -->"


def markdown():
    """The generated block for AlpogeSixSphere.md."""
    out = [BEGIN, "", "| Part | check | kind | rests on |", "| --- | --- | --- | --- |"]
    part = {s: p for p, s, _, _ in SOURCES}
    for key, v in KIND.items():
        notes = []
        if key in QUOTED:
            notes.append("quoted: " + QUOTED[key])
        if key in SECOND_ROUTE:
            notes.append("second route: " + SECOND_ROUTE[key])
        if key in LEAN:
            notes.append("Lean: " + ", ".join("`%s`" % n for n in LEAN[key]))
        if key in CRITICAL:
            notes.append("**Theorem XI**")
        name = key[1].replace("|", "\\|")
        out.append("| %s | `%s` | %s | %s |" % (part[key[0]], name,
                                               ", ".join(_NAMES[c] for c in v),
                                               "; ".join(notes)))
    t = counts()["total"]
    out += ["", "%d checks: %s." % (len(KIND), ", ".join(
        "%d %s" % (t[k], _NAMES[k]) for k in VOCABULARY if t[k])),
        "%d checks are linked to %d Lean facts (`sixsphere_lean.JACOBI_FACTS`, "
        "kernel-checked by `status_jacobi()`)." % (
            len(LEAN), len({n for v in LEAN.values() for n in v})),
        "Quoted inputs on Theorem XI's critical path: %s." % "; ".join(critical_quoted()),
        "", END]
    return "\n".join(out)


def write_markdown(path=None):
    """Replace the generated block in AlpogeSixSphere.md; returns True if it changed."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "AlpogeSixSphere.md")
    text = open(path, encoding="utf-8").read()
    i, j = text.index(BEGIN), text.index(END) + len(END)
    new = text[:i] + markdown() + text[j:]
    if new != text:
        open(path, "w", encoding="utf-8").write(new)
    return new != text


def status_ledger(rows=None):
    st = selftest(rows)
    for name, c in st.items():
        print("  {:<62} {}".format(name, "ok" if c else "FAIL"))
    cq = critical_quoted()
    ok = len(cq) == 2 and cq == sorted([_SL2Z, _THETA])
    print("  {:<62} {}".format("Theorem XI quoted inputs: SL(2,Z) generators, theta laws",
                                "ok" if ok else "FAIL"))
    return all(st.values()) and ok


if __name__ == "__main__":
    import sys
    rows = collect()
    if "--write" in sys.argv:
        print("changed" if write_markdown() else "unchanged")
    c = counts()
    for p, *_ in SOURCES:
        print("  %-5s %s" % (p, " ".join("%s%d" % (k, c[p][k]) for k in VOCABULARY)))
    sys.exit(0 if status_ledger(rows) else 1)
