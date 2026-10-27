r"""
pyCICY.sixsphere_classes -- the four near-miss classes are one group up to
isogeny (Part XII).

Part VII found that the flag-class lattice data for (3, 4, inf) falls into
four GL(4, Z) classes, and Part IX that they are told apart by the
divisibility d in {1, 2, 3, 6} of Q(gamma, .), with every reachable p a
multiple of d and least |p| = d (observed in a box). Part VII declined the
analytic side: whether the three near misses carry period functions at all
was "not a lattice question". This module answers both questions.

1. **Normal form** (:func:`adapted_forms`). For each class an explicit
   unimodular P_d (stored, re-verified) brings the invariant form to

       P_d^t Q P_d = d E_03 + (6/d) E_12          (E_ij = e_i e_j^t - e_j e_i^t),

   and fixes gamma. The paper's class is d = 1 after the swap; for every d
   the Pfaffian is d * (6/d) = 6.
2. **Isogeny** (:func:`isogeny`). With D_d = diag(d, 1, 1, d), the matrices

       W_i = D_d (P_d^-1 T_i P_d) D_d^-1

   are *the same two matrices for all four classes*, they preserve Q0 and
   fix gamma, and h(0, 1, 0) conjugates them onto the paper's (T1, T2)
   exactly. So each class is the paper's monodromy group acting on the
   sublattice L_d = D_d Z^4, of index d^2 and stable under the group: a
   fibrewise isogeny of degree d^2 of the same torus fibration. Over Q all
   four classes are one class, conjugated by C_d = P_d D_d^-1 h(0,1,0)^-1.
3. **Structure group and index 12** (:func:`structure_isomorphism`,
   :func:`index_proof`). Phi_d(g) = D_d g D_d^-1 is an isomorphism from the
   integral automorphisms of (Q_d, gamma) onto the integral Jacobi group
   G(Z) of Part IX: symbolically, D^-1 g D is integral for every g in G(Z),
   since 6/d is an integer; conversely an integral automorphism of
   (Q_d, gamma) has translation column in dZ^2 (searched in a box). Since
   h(0,1,0) W_i h(0,1,0)^-1 = T_i and ker chi is normal, <W1, W2> = ker chi;
   pulled back, each class's monodromy group is ker(chi o Phi_d), of index
   12. Part IX's word-search route, which needs no isogeny, gives the
   central index 12 independently (:func:`central_index_route`, slow).
4. **Exact p** (:func:`reachable_p`). Freeness of a twist v is periodic in
   v mod m F (F the fixed lattice, m the order), so a finite enumeration of
   (Z/m)^rank F gives every reachable gamma(v), and p = 12 l0 - 4a - 3b
   then covers whole residue classes mod 12. Result, exactly rather than in
   a box: the gcd of the reachable p is d and the least nonzero |p| is d.
5. **Periods** (:func:`laws_d`, :func:`theta_check`). Because Pi_d = Pi D^-1
   up to rescaling a row, class d has period matrix

       Z_d = [[6 mu / d^2, tau], [beta, mu]]       (mu = d * mu_paper),

   and :func:`laws_d` derives each class's laws from the integer matrices
   alone, as :func:`pyCICY.sixsphere.laws_from_lattice` does for the paper.
   The near misses therefore do carry period data: the isogenous variation.
   Their theta function is the pullback F_d(tau, mu, beta) = F(tau, mu/d,
   beta), with multiplier chi o Phi_d on the whole structure group
   (numerical, 25 digits).
6. **Zeros: a conjecture refuted** (:func:`zero_divisor_d`,
   :func:`characteristic_control`). The next-steps notes conjectured that the
   class-d theta vanishes at d-torsion points. It does not. F_d vanishes on
   mu in d(Z + tau Z), the orbit of the origin under the class's own
   translations, and is nonzero at mu = 1, tau, 1 + tau (the d-torsion of
   C / d(Z + tau Z)) for d > 1. Control: of the d^2 theta functions with
   characteristics (1/2 + a/d, 1/2 + b/d) in mu/d, only a = b = 0 is
   invariant under the class's T1, T2; and e(beta/12) theta_1(tau, mu)/eta,
   whose zeros are exactly those d-torsion points, is not invariant for
   d > 1.

What is proved here: 1, 2 and 3 (exact integer arithmetic plus a symbolic
integrality argument; the converse direction of the isomorphism is a box
search), 4 (exact, by the periodicity argument), the laws in 5 (sympy).
Measured: the multipliers in 5 and the zero values in 6 (mpmath).
What is not claimed: that the near-miss data builds a complex S^6. The
isogenous period map exists, but Alpoge's |p| = 1 condition is what the
fundamental group needs, and only d = 1 reaches it.
"""

import itertools

import mpmath as mp
from sympy import (Matrix, Rational, cancel, diag, eye, lambdify, linsolve,
                   symbols, zeros)

from . import sixsphere as S
from . import sixsphere_gauge as G
from . import sixsphere_search as SR
from . import sixsphere_theta as TH

__all__ = ["CLASS_REPS", "ADAPTED", "W1", "W2", "CONJUGATOR", "D", "Q_adapted",
           "adapted_forms", "isogeny", "rational_conjugators",
           "structure_isomorphism", "index_proof", "central_index_route",
           "reachable_p", "period_matrix_d", "laws_d", "F_d", "theta_check",
           "zero_divisor_d", "theta_char", "characteristic_control",
           "status_classes"]

#: One representative (T1, T2) per class, from ``funnel(3)`` + ``classify``,
#: keyed by the divisibility d of Q(gamma, .). d = 1 is the paper's class.
CLASS_REPS = {
    1: (Matrix([[1, -3, 0, -2], [0, -1, 1, -2], [0, -1, 0, 0], [0, 0, 0, 1]]),
        Matrix([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 1], [0, 0, 0, 1]])),
    2: (Matrix([[1, 0, 0, 0], [0, -1, 1, -1], [0, -1, 0, 0], [0, 0, 0, 1]]),
        Matrix([[1, 0, 1, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])),
    3: (Matrix([[1, -1, 0, -1], [0, -1, 1, -3], [0, -1, 0, 0], [0, 0, 0, 1]]),
        Matrix([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 1], [0, 0, 0, 1]])),
    6: (Matrix([[1, 1, 0, 1], [0, -1, 1, -3], [0, -1, 0, 0], [0, 0, 0, 1]]),
        Matrix([[1, 0, 1, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])),
}

#: The adapted bases: P_d^t Q P_d = Q_adapted(d), P_d gamma = gamma.
ADAPTED = {
    1: Matrix([[1, -3, -3, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
    2: Matrix([[1, -2, -2, 0], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, -1]]),
    3: Matrix([[1, -1, -1, 0], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, -1]]),
    6: Matrix([[1, -1, -1, 0], [0, 1, 0, -3], [0, 0, 1, 3], [0, 0, 0, -1]]),
}

#: The common image of every class: D_d P_d^-1 T_i P_d D_d^-1.
W1 = Matrix([[1, -12, 0, 8], [0, -1, 1, 2], [0, -1, 0, 0], [0, 0, 0, 1]])
W2 = Matrix([[1, 0, -6, -3], [0, 0, -1, 0], [0, 1, 0, -1], [0, 0, 0, 1]])

#: h W_i h^-1 = T_i (the paper's matrices), with h = h(0, 1, 0).
CONJUGATOR = G.heis(0, 1, 0)

DS = (1, 2, 3, 6)


def D(d):
    return diag(d, 1, 1, d)


def Q_adapted(d):
    """d E_03 + (6/d) E_12."""
    Q = zeros(4)
    Q[0, 3], Q[3, 0] = d, -d
    Q[1, 2], Q[2, 1] = 6 // d, -(6 // d)
    return Q


def _gamma_fixed(T):
    return Matrix(T)[:, 0] == Matrix([1, 0, 0, 0])


# --------------------------------------------------------------------------
# 1, 2. normal form and isogeny
# --------------------------------------------------------------------------

def adapted_forms():
    """Per class: d from Q(gamma, .), Pf 6, P_d unimodular, the normal form."""
    out = {}
    for d in DS:
        T1, T2 = CLASS_REPS[d]
        dd, pf, Q = G._divisibility(T1, T2)
        P = ADAPTED[d]
        out[d] = {"d": dd, "pfaffian": pf, "det P": P.det(),
                  "normal form": P.T * Q * P == Q_adapted(d),
                  "P fixes gamma": _gamma_fixed(P),
                  "ok": (dd == d and pf == 6 and abs(P.det()) == 1
                         and P.T * Q * P == Q_adapted(d) and _gamma_fixed(P))}
    return out


def isogeny():
    """U_i = P^-1 T_i P preserves Q_adapted(d); W_i = D U_i D^-1 is the same
    for every d, preserves Q0, fixes gamma, and is h-conjugate to the paper's
    T_i. L_d = D Z^4 has index d^2 and is <W1, W2>-stable."""
    out = {}
    for d in DS:
        P, Dd = ADAPTED[d], D(d)
        Us = [P.inv() * T * P for T in CLASS_REPS[d]]
        Ws = [Dd * U * Dd.inv() for U in Us]
        out[d] = {
            "U integral": all(x.is_integer for U in Us for x in U),
            "U preserve Q_d": all(U.T * Q_adapted(d) * U == Q_adapted(d)
                                  for U in Us),
            "W = (W1, W2)": Ws == [W1, W2],
            "index of L_d": abs(Dd.det()),
            # L_d stable: W (D Z^4) = D U Z^4 within D Z^4, as U is integral
            "L_d stable": all(x.is_integer for W in Ws
                              for x in Dd.inv() * W * Dd)}
        out[d]["ok"] = (out[d]["U integral"] and out[d]["U preserve Q_d"]
                        and out[d]["W = (W1, W2)"] and out[d]["L_d stable"]
                        and out[d]["index of L_d"] == d * d)
    common = {
        "W preserve Q0": all(W.T * S.Q0 * W == S.Q0 for W in (W1, W2)),
        "W fix gamma": all(_gamma_fixed(W) for W in (W1, W2)),
        "h W h^-1 = paper T": (CONJUGATOR * W1 * CONJUGATOR.inv() == S.T1
                               and CONJUGATOR * W2 * CONJUGATOR.inv() == S.T2)}
    return {"classes": out, "common": common,
            "ok": all(v["ok"] for v in out.values()) and all(common.values())}


def rational_conjugators():
    """C_d = P_d D_d^-1 h^-1: C_d T_i C_d^-1 = class-d T_i, det = +-1/d^2."""
    out = {}
    for d in DS:
        C = ADAPTED[d] * D(d).inv() * CONJUGATOR.inv()
        out[d] = {"det": C.det(),
                  "conjugates": all(C * Tp * C.inv() == T for Tp, T in
                                    zip((S.T1, S.T2), CLASS_REPS[d]))}
    return out


# --------------------------------------------------------------------------
# 3. the structure group and the index
# --------------------------------------------------------------------------

def structure_isomorphism(box=3):
    """Phi_d(g) = D g D^-1 from Aut(Q_d, gamma)(Z) onto G(Z).

    Forward (symbolic): for g = L(M) h(c1, c2, s) with integer entries,
    D^-1 g D has entries in Z[M, c, s, 6/d], hence integral. It preserves
    Q_d because D^-t Q_d D^-1 = Q0 / d. Backward (box): every integral g
    with gamma fixed, row 3 = e_3, M in SL(2, Z) and translation column t,
    preserving Q_d, has t in d Z^2 -- so D g D^-1 is integral."""
    a, b, c, dd, c1, c2, s = symbols("a b c dd c1 c2 s", integer=True)
    M = Matrix([[a, b], [c, dd]])
    g = G.levi(M) * G.heis(c1, c2, s)
    out = {}
    for d in DS:
        h = D(d).inv() * g * D(d)
        poly = all(all(Rational(cf).q == 1 for cf in
                       x.as_poly(a, b, c, dd, c1, c2, s).coeffs())
                   for x in h if x != 0)
        scal = D(d).inv().T * Q_adapted(d) * D(d).inv() == S.Q0 / d
        # backward: solve the top row from Q_d-invariance, test t in dZ^2
        sl2 = [Matrix([[p, q], [u, v]]) for p, q, u, v in
               itertools.product(range(-1, 2), repeat=4) if p * v - q * u == 1]
        back_ok, tried = True, 0
        r0, r1, s0 = symbols("r0 r1 s0")
        for Mm in sl2:
            for t in itertools.product(range(-box, box + 1), repeat=2):
                gg = Matrix([[1, r0, r1, s0], [0, Mm[0, 0], Mm[0, 1], t[0]],
                             [0, Mm[1, 0], Mm[1, 1], t[1]], [0, 0, 0, 1]])
                eqs = list(gg.T * Q_adapted(d) * gg - Q_adapted(d))
                sol = list(linsolve(eqs, [r0, r1]))
                if not sol:
                    continue
                rr = sol[0]
                integral = all(Rational(x).q == 1 for x in rr)
                tried += 1
                in_dZ = t[0] % d == 0 and t[1] % d == 0
                if integral != in_dZ:
                    back_ok = False
        out[d] = {"D^-1 G(Z) D integral": poly, "D^-t Q_d D^-1 = Q0/d": scal,
                  "integral iff t in dZ^2": back_ok, "tried": tried}
    return out


def index_proof():
    """<W1, W2> = ker chi in G(Z), hence index 12 for every class.

    chi(W1) = chi(W2) = 0, and h(0,1,0) W_i h(0,1,0)^-1 = T_i with
    <T1, T2> = ker chi (Part IX). ker chi is normal, so <W1, W2> =
    h^-1 ker chi h = ker chi. Phi_d carries class d's group onto <W1, W2>
    and Aut(Q_d, gamma)(Z) onto G(Z), so the class-d group is
    ker(chi o Phi_d), of index 12."""
    M1, M2 = W1[1:3, 1:3], W2[1:3, 1:3]
    hi = G.holonomy_index()
    out = {"chi(W1)": TH.chi(W1), "chi(W2)": TH.chi(W2),
           "M1 M2 = T": M1 * M2 == G._TT, "M2 = S": M2 == G._ST,
           "conjugate to paper": isogeny()["common"]["h W h^-1 = paper T"],
           "paper index 12 (Part IX)": hi["proved"]}
    out["proved"] = (out["chi(W1)"] == 0 and out["chi(W2)"] == 0
                     and out["conjugate to paper"] and hi["proved"])
    return out


def central_index_route(max_elements=700000):
    """Independent of the isogeny: Part IX's word search on <W1, W2> (the
    same gcd as on each class rep, since P_d fixes gamma and e_3^* up to
    sign). Slow, about a minute and a half."""
    return G.central_index(W1, W2, max_elements=max_elements)


# --------------------------------------------------------------------------
# 4. the reachable p, exactly
# --------------------------------------------------------------------------

def _reachable_gamma(T, m):
    """gamma(v) over free twists v, as residues mod m g: freeness depends on
    v mod m F only (a translation by an integer vector is trivial), so
    co in (Z/m)^rank F covers every case."""
    A = Matrix(T).inv().T
    F = S.int_kernel(A - eye(4))
    g = 0
    for x in F[:, 0]:
        g = G.gcd(g, int(x))
    mod = m * g if g else 0
    vals = set()
    for co in itertools.product(range(m), repeat=F.rows):
        v = zeros(4, 1)
        for i, cf in enumerate(co):
            v += cf * F.row(i).T
        if SR._free(A, m, v):
            vals.add(int(v[0]) % mod if mod else int(v[0]))
    return vals, mod


def reachable_p(d=None, T=None):
    """Exact set of reachable p mod 12, its gcd with 12, least nonzero |p|."""
    T1, T2 = CLASS_REPS[d] if T is None else T
    a, ma = _reachable_gamma(T1, 3)
    b, mb = _reachable_gamma(T2, 4)
    # a ranges over a0 + ma Z, -4a over -4a0 + 4 ma Z, and 4 ma, 3 mb are
    # multiples of 12 (ma in 3Z, mb in 4Z), absorbed by 12 l0
    res = sorted({(-4 * x - 3 * y) % 12 for x in a for y in b})
    g = 12
    for r in res:
        g = G.gcd(g, r)
    least = min(min(r, 12 - r) if r else 12 for r in res)
    return {"residues mod 12": res, "gcd": g, "least |p|": least,
            "moduli": (ma, mb), "ma in 3Z, mb in 4Z": ma % 3 == 0 and mb % 4 == 0}


# --------------------------------------------------------------------------
# 5. periods and theta
# --------------------------------------------------------------------------

def period_matrix_d(d):
    """Pi_d = [Z_d | I], Z_d = [[6 mu / d^2, tau], [beta, mu]]."""
    t, m, b = S.TAU, S.MU, S.BETA
    return Matrix([[6 * m / d ** 2, t, 1, 0], [b, m, 0, 1]])


def laws_d(T, d):
    """The law of T (in the adapted basis of class d) from the matrix alone,
    with the consistency condition Z_d[0, 0] = (6/d^2) Z_d[1, 1]."""
    Pm = period_matrix_d(d) * Matrix(T).T
    R = Pm[:, 2:4].inv().applyfunc(cancel)
    Z = (R * Pm[:, 0:2]).applyfunc(cancel)
    return {"shape_ok": cancel(Z[0, 0] - Rational(6, d * d) * Z[1, 1]) == 0,
            "law": (Z[0, 1], Z[1, 1], Z[1, 0])}


def _act_d(g, d, x=TH.BASE):
    law = laws_d(g, d)["law"]
    return lambdify((S.TAU, S.MU, S.BETA), law, "mpmath")(*x)


def F_d(d):
    """F pulled back along the isogeny: F(tau, mu/d, beta)."""
    return lambda t, m, b: TH.F(t, m / d, b)


def _base_d(d):
    t, m, b = TH.BASE
    return (t, d * m, b)


def theta_check(tol=1e-12):
    """Per class: laws have the Z_d shape; F_d is fixed by U1, U2; on
    Phi_d^-1 of the five generators of G(Z) its multiplier is chi."""
    gens = {"L(S)": G.levi(G._ST), "L(T)": G.levi(G._TT),
            "h(1,0,0)": G.heis(1, 0, 0), "h(0,1,0)": G.heis(0, 1, 0),
            "z": G.heis(0, 0, 1)}
    out = {}
    for d in DS:
        f, x0 = F_d(d), _base_d(d)
        Us = [ADAPTED[d].inv() * T * ADAPTED[d] for T in CLASS_REPS[d]]
        mats = {"U1": Us[0], "U2": Us[1]}
        mats.update({k: D(d).inv() * g * D(d) for k, g in gens.items()})
        rows = {}
        for k, g in mats.items():
            r = f(*_act_d(g, d, x0)) / f(*x0)
            want = 0 if k in ("U1", "U2") else TH.chi(D(d) * g * D(d).inv())
            rows[k] = {"shape_ok": laws_d(g, d)["shape_ok"],
                       "ok": abs(r - TH.e(mp.mpf(want) / 12)) < tol}
        out[d] = {"rows": rows,
                  "ok": all(v["shape_ok"] and v["ok"] for v in rows.values())}
    return out


# --------------------------------------------------------------------------
# 6. zeros, and the characteristic control
# --------------------------------------------------------------------------

def zero_divisor_d(d):
    """|F_d| on d(Z + tau Z) (zero) and at the d-torsion 1, tau, 1 + tau of
    C / d(Z + tau Z) (nonzero for d > 1)."""
    t, _, b = TH.BASE
    f = F_d(d)
    orbit = max(abs(f(t, d * (m + n * t), b)) for m in (-1, 0, 1)
                for n in (-1, 0, 1))
    tors = min(abs(f(t, u + v * t, b)) for u, v in ((1, 0), (0, 1), (1, 1)))
    return {"max on d Lambda": float(orbit), "min at 1, tau, 1+tau": float(tors)}


def theta_char(a, b, t, z):
    """theta[a, b](tau, z) = sum_n e((n + a)^2 tau / 2 + (n + a)(z + b));
    theta[1/2, 1/2] = -theta_1."""
    return mp.nsum(lambda n: TH.e((n + a) ** 2 * t / 2 + (n + a) * (z + b)),
                   [-mp.inf, mp.inf])


def characteristic_control(d, tol=1e-12):
    """Which e(beta/12) theta[1/2 + i/d, 1/2 + j/d](tau, mu/d) / eta are fixed
    by class d's U1, U2? And is the naive e(beta/12) theta_1(tau, mu)/eta
    (zeros at the d-torsion) fixed?"""
    x0 = _base_d(d)
    Us = [ADAPTED[d].inv() * T * ADAPTED[d] for T in CLASS_REPS[d]]
    ys = [_act_d(U, d, x0) for U in Us]
    half = mp.mpf(1) / 2
    surv = []
    for i, j in itertools.product(range(d), repeat=2):
        def f(t, m, b, i=i, j=j):
            return (TH.e(b / 12) * theta_char(half + mp.mpf(i) / d,
                                              half + mp.mpf(j) / d, t, m / d)
                    / TH.eta(t))
        f0 = f(*x0)
        if all(abs(f(*y) / f0 - 1) < tol for y in ys):
            surv.append((i, j))

    def naive(t, m, b):
        return TH.F(t, m, b)
    naive_fixed = all(abs(naive(*y) / naive(*x0) - 1) < tol for y in ys)
    return {"candidates": d * d, "survivors": surv, "naive fixed": naive_fixed}


def status_classes(slow=False):
    """Run every check of Part XII and print a summary; True if all pass."""
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    af = adapted_forms()
    ok("normal form d E03 + (6/d) E12, Pf 6, P unimodular, 4 classes",
       all(v["ok"] for v in af.values()))
    iso = isogeny()
    ok("D P^-1 T_i P D^-1 = (W1, W2) for all four classes", iso["ok"])
    ok("h(0,1,0) conjugates (W1, W2) onto the paper's (T1, T2)",
       iso["common"]["h W h^-1 = paper T"])
    rc = rational_conjugators()
    ok("Q-conjugate to the paper by C_d, det +-1/d^2",
       all(v["conjugates"] and abs(v["det"]) == Rational(1, d * d)
           for d, v in rc.items()))
    si = structure_isomorphism()
    ok("Phi_d: Aut(Q_d, gamma)(Z) = G(Z) (symbolic + box)",
       all(v["D^-1 G(Z) D integral"] and v["D^-t Q_d D^-1 = Q0/d"]
           and v["integral iff t in dZ^2"] for v in si.values()))
    ip = index_proof()
    ok("<W1, W2> = ker chi: index 12 for every class", ip["proved"])
    rp = {d: reachable_p(d) for d in DS}
    ok("exact: gcd of reachable p = d, least |p| = d",
       all(v["gcd"] == d and v["least |p|"] == d for d, v in rp.items()))
    tc = theta_check()
    ok("class laws have the Z_d shape; F_d multiplier = chi o Phi_d",
       all(v["ok"] for v in tc.values()))
    zd = {d: zero_divisor_d(d) for d in DS}
    ok("F_d = 0 on d(Z + tau Z); nonzero at 1, tau, 1+tau for d > 1",
       all(v["max on d Lambda"] < 1e-15 for v in zd.values())
       and all(zd[d]["min at 1, tau, 1+tau"] > 1e-3 for d in (2, 3, 6)))
    cc = {d: characteristic_control(d) for d in (2, 3, 6)}
    ok("control: only characteristic (0, 0) survives; naive theta fails",
       all(v["survivors"] == [(0, 0)] and not v["naive fixed"]
           for v in cc.values()))
    if slow:
        ok("word search: central index of <W1, W2> is 12",
           central_index_route()["gcd"] == 12)
    for name, c in checks:
        print("  {:<62} {}".format(name, "ok" if c else "FAIL"))
    return all(c for _, c in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if status_classes() else 1)
