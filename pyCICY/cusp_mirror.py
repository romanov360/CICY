r"""
pyCICY.cusp_mirror -- the cusp substrate in a field against the quantised
dP_6 hexagon (Part XIV).

Lead 8 asked whether the substrate's Hofstadter spectrum (Part X) is a
polynomial function of the quantised mirror curve of local B_3 = dP_6, whose
Newton polygon is the hexagon and whose quantisation is the triangular
lattice in a field (Hatsuda, Sugimoto and Xu, arXiv:1701.01561, extending
Hatsuda, Katsura and Tachikawa, arXiv:1606.01894). Part X's zero-field
characteristic polynomial,

    det(E - H) = E^6 - (12 + s) E^4 - 3 (s - 6) E^2 - Delta Delta-bar,

has s = x + 1/x + y + 1/y + x/y + y/x, the hexagon itself, in every
coefficient but the last. The answer in that form is no; what holds instead
is an exact operator identity in which the quantised hexagon appears
literally.

1. **Zero field needs two hexagons** (:func:`zero_field_invariants`).
   Exactly (sympy),

       Delta Delta-bar = 12 - s^2 + 4 u,
       u = x y + 1/(x y) + x^2/y + y/x^2 + y^2/x + x/y^2,

   and u is the *long-root* hexagon: Newton polygon with vertices
   (1,1), (2,-1), (1,-2), ..., which is the dP_6 hexagon on the index-3
   sublattice spanned by (1,1) and (2,-1). The 3-torsion V sites put it
   there. s and u are algebraically independent (their Jacobian is
   nonzero), so even at zero field the spectrum is a function of the pair
   (s, u) and not of s.
2. **The exact identity in a field** (:func:`square_identity`). The graph
   is bipartite, E sites against C and V sites, so H^2 preserves C u V. On
   an open patch, with Peierls phases from the straight-line Landau gauge
   of :func:`pyCICY.cusp_hofstadter.landau_phase`, for every kite flux phi:

       H^2 |_CC = 6 + S ,     S = straight-line Peierls hops C -> C + h,
                               h in the six short roots (the hexagon),
       H^2 |_VV = 3 + H_V ,   H_V = the V honeycomb, straight-line Peierls,
       H^2 |_CV = [2]_q K ,   [2]_q = q^{1/2} + q^{-1/2} = 2 cos(pi phi),
                               K = straight-line Peierls hops C -> V,
                               q = e^{2 pi i phi}.

   The kite-flux residue of each two-step path is what turns the C-V
   amplitude into the quantum integer [2]_q.
3. **S is the quantised hexagon** (:func:`hexagon_match`). On the C lattice,
   S is :class:`pyCICY.quantum_curve.QuantumCurve` for B_3 at flux
   Phi = 6 phi mod 2 (a C triangle carries half a cell, so the period is
   2, not 1). This is checked on the gauge-invariant moments
   tr S^n / dim, averaged exactly over the Brillouin zone, n <= 10.
4. **phi = 1/2 decouples** (:func:`half_flux`). [2]_q = 0, so C and V
   separate: spec H^2 = spec(6 + S) u spec(3 + H_V), with S the pi-flux
   triangular lattice (Phi = 1), whose symbol is
   2 cos k1 + 2 cos k2 - 2 cos(k1 - k2) on [-6, 3]. Both blocks fill
   [0, 9] and [0, 6], so the substrate spectrum is gapless on [-3, 3].
5. **The Chern number is in the coupling** (:func:`chern_obstruction`). At
   phi = 1/6 and 1/3 the flux per cell is an integer: the algebra generated
   by S and the long-root operator U is commutative ([S, U] = 0, as against
   phi = 1/10 where it is not), and the V hexagon encloses integer flux.
   Every spectral projection of a function of S and U is a sum of plane-wave
   line bundles, Chern 0. Yet Part X and Part XIII find substrate gaps with
   Chern +-1 at these fluxes. So no spectral projection of the substrate is
   a function of the quantised dP_6 curve at the same flux, and the Chern
   number is carried by the [2]_q-weighted C-V coupling.
6. **Control** (:func:`naive_quantisation`). Replacing s and u in the
   zero-field characteristic polynomial by the Weyl-quantised S and U and
   taking a determinant reproduces det(E - H) at zero field (1e-12), and
   fails by many orders of magnitude at phi = 1/6, 1/12, 1/9.

What this is: an exact local identity (residual 1e-15 on a patch and on the
magnetic supercell), a moment-by-moment match with the quantised hexagon, and
the falsification of lead 8 as stated, with the reason. What it is not: a
derivation of the substrate's gap Chern numbers from the topological string,
or a statement about the spectral problem of the full two-hexagon curve.
"""

from fractions import Fraction as Fr
from math import pi, sqrt

import numpy as np
from sympy import expand as _expand, simplify as _simplify, \
    symbols as _symbols, Matrix as _Matrix

from . import cusp_hofstadter as C
from . import quantum_curve as Q

__all__ = [
    "SHORT", "LONG", "zero_field_invariants", "c_operator", "square_identity",
    "supercell_identity", "hexagon_match", "half_flux", "chern_obstruction",
    "naive_quantisation", "status_mirror",
]

#: The short roots (the dP_6 hexagon) and the long roots (the same hexagon on
#: the index-3 sublattice), as C-lattice hops.
SHORT = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
LONG = [(1, 1), (2, -1), (1, -2), (-1, -1), (-2, 1), (-1, 2)]

_C_IDX = [i for i, k in enumerate(C.SITE_KIND) if k == "C"]
_V_IDX = [i for i, k in enumerate(C.SITE_KIND) if k == "V"]
_E_IDX = [i for i, k in enumerate(C.SITE_KIND) if k == "E"]


# ---------------------------------------------------------------------------
# 1. zero field
# ---------------------------------------------------------------------------

def zero_field_invariants():
    r"""Delta Delta-bar in terms of the two hexagons, and the charpoly in (s, u)."""
    x, y, E = _symbols("x y E")
    s = x + 1 / x + y + 1 / y + x / y + y / x
    u = sum(x ** a * y ** b for a, b in LONG)
    Delta = (1 - x) * (1 - y) * (1 - x / y)
    Delta_bar = (1 - 1 / x) * (1 - 1 / y) * (1 - y / x)
    identity = _simplify(_expand(Delta * Delta_bar - (12 - s ** 2 + 4 * u))) == 0
    p = C.bloch_charpoly()["charpoly"]
    form = E ** 6 - (12 + s) * E ** 4 - 3 * (s - 6) * E ** 2 \
        - (12 - s ** 2 + 4 * u)
    charpoly_ok = _simplify(_expand(p - form)) == 0
    # index of the sublattice spanned by the long roots
    index = abs(LONG[0][0] * LONG[1][1] - LONG[0][1] * LONG[1][0])
    # independence: Jacobian of (s, u) in (k1, k2) at a generic point
    k1, k2 = 0.37, 1.21

    def grad(vecs):
        return np.array([sum(-2 * a * np.sin(a * k1 + b * k2) for a, b in vecs[:3]),
                         sum(-2 * b * np.sin(a * k1 + b * k2) for a, b in vecs[:3])])
    jac = float(np.linalg.det(np.array([grad(SHORT), grad(LONG)])))
    return {"DeltaDeltabar_is_12_minus_s2_plus_4u": identity,
            "charpoly_in_s_u": charpoly_ok,
            "long_root_index": index,
            "long_is_short_on_sublattice": _long_is_image(),
            "jacobian_s_u": jac}


def _long_is_image():
    """LONG is the image of SHORT under the index-3 map (1,0)->(1,1), (0,1)->(-1,2)
    up to the order of the hexagon."""
    img = sorted((a - b, a + 2 * b) for a, b in SHORT)
    return img == sorted(LONG)


# ---------------------------------------------------------------------------
# 2. the identity on a patch and on the supercell
# ---------------------------------------------------------------------------

def c_operator(phi, k1, k2, vecs=SHORT):
    r"""Straight-line Peierls hops C -> C + h, h in ``vecs``, on the q x 1
    C supercell at kite flux phi (same gauge and cell as
    :func:`pyCICY.cusp_hofstadter.magnetic_bloch`)."""
    phi = Fr(phi)
    q = phi.denominator
    O = np.zeros((q, q), complex)
    for m in range(q):
        for d in vecs:
            mb = (m + d[0]) % q
            S1 = (m + d[0] - mb) // q
            th = C.landau_phase((m, 0), (m + d[0], d[1]), phi)
            O[m, mb] += np.exp(1j * (th + k1 * S1 + k2 * d[1]))
    return O


def _kind(p):
    u, v = p[0] % 1, p[1] % 1
    if u == 0 and v == 0:
        return "C"
    if (u, v) in ((Fr(1, 3), Fr(1, 3)), (Fr(2, 3), Fr(2, 3))):
        return "V"
    return "E"


def square_identity(phi, radius=12.0, inner=6.0):
    r"""Largest residual of each block of H^2 against the identity, on an open
    patch, over rows within ``inner`` of the centre (away from the edge).

    Returns {block: residual} for CC, VV, CV, EC u EV (must vanish), and the
    diagonal values per site kind.
    """
    phi = Fr(phi)
    pos, bonds = C.finite_patch(radius)
    H = C.finite_hamiltonian(pos, bonds, phi)
    H2 = H @ H
    XY = C.to_xy(np.array([[float(a), float(b)] for a, b in pos]))
    r = np.hypot(XY[:, 0], XY[:, 1])
    K = [_kind(p) for p in pos]
    q2 = 2 * np.cos(pi * float(phi))
    res = {"CC": 0.0, "VV": 0.0, "CV": 0.0, "E-CV": 0.0}
    diag = {"C": set(), "E": set(), "V": set()}
    for i in range(len(pos)):
        if r[i] > inner:
            continue
        diag[K[i]].add(round(H2[i, i].real, 9))
        for j in range(len(pos)):
            if i == j:
                continue
            d = np.linalg.norm(XY[i] - XY[j])
            kk = K[i] + K[j]
            if "E" in kk and kk != "EE":
                res["E-CV"] = max(res["E-CV"], abs(H2[i, j]))
                continue
            if kk == "EE":
                continue
            straight = np.exp(-1j * C.landau_phase(pos[i], pos[j], phi))
            if kk == "CC":
                pred = straight if abs(d - 2 * sqrt(3)) < 1e-9 else 0
                res["CC"] = max(res["CC"], abs(H2[i, j] - pred))
            elif kk == "VV":
                pred = straight if abs(d - 2) < 1e-9 else 0
                res["VV"] = max(res["VV"], abs(H2[i, j] - pred))
            else:
                pred = q2 * straight if abs(d - 2) < 1e-9 else 0
                res["CV"] = max(res["CV"], abs(H2[i, j] - pred))
    return {"residual": res,
            "diag": {k: sorted(v) for k, v in diag.items()}}


def _blocks(phi, k1, k2):
    H = C.magnetic_bloch(phi, k1, k2)
    n = len(C.SITES)
    q = Fr(phi).denominator
    Ci = [a + n * m for m in range(q) for a in _C_IDX]
    Vi = [a + n * m for m in range(q) for a in _V_IDX]
    H2 = H @ H
    return H2, Ci, Vi


def supercell_identity(phis=(Fr(1, 2), Fr(1, 3), Fr(1, 5), Fr(2, 7)), seed=0):
    r"""On the magnetic supercell: max |H^2_CC - (6 + S)|, with S built by
    :func:`c_operator` (S or its transpose, whichever the supercell's
    orientation gives), over random twists."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for phi in phis:
        for _ in range(3):
            k1, k2 = rng.uniform(0, 2 * pi, 2)
            H2, Ci, _ = _blocks(phi, k1, k2)
            S = c_operator(phi, k1, k2)
            B = H2[np.ix_(Ci, Ci)] - 6 * np.eye(len(Ci))
            worst = max(worst, min(np.max(np.abs(B - S)),
                                   np.max(np.abs(B - S.T))))
    return worst


# ---------------------------------------------------------------------------
# 3. S is the quantised hexagon
# ---------------------------------------------------------------------------

def _moments(fn, nmax, nk):
    acc = np.zeros(nmax + 1)
    ks = 2 * pi * np.arange(nk) / nk
    for a in ks:
        for b in ks:
            ev = np.linalg.eigvalsh(fn(a, b))
            acc += [np.mean(ev ** n) for n in range(nmax + 1)]
    return acc / nk ** 2


def hexagon_match(phi, nmax=10, nk=13):
    r"""Relative difference of tr S^n / dim and the B_3 quantum curve's at
    Phi = 6 phi mod 2, n <= nmax. ``nk > nmax`` makes the grid average exact
    (the moments are trigonometric polynomials of degree <= nmax)."""
    phi = Fr(phi)
    Phi = (6 * phi) % 2
    B3 = Q.QuantumCurve.from_polygon("B3")
    p, q = Phi.numerator, Phi.denominator
    mS = _moments(lambda a, b: c_operator(phi, a, b), nmax, nk)
    mB = _moments(lambda a, b: B3.bloch_matrix(p, q, a, b), nmax, nk)
    return {"Phi": Phi, "points": sorted(B3.points),
            "rel_diff": float(np.max(np.abs(mS - mB) / np.maximum(1, np.abs(mS)))),
            "m3": float(mS[3])}


# ---------------------------------------------------------------------------
# 4. phi = 1/2
# ---------------------------------------------------------------------------

def half_flux(nk=12):
    r"""The C-V block of H^2 vanishes at phi = 1/2; S is pi-flux triangular;
    spec H^2 = spec(6 + S) u spec(3 + H_V); the spectrum fills [-3, 3]."""
    phi = Fr(1, 2)
    cv, sym, split = 0.0, 0.0, 0.0
    allE = []
    ks = 2 * pi * np.arange(nk + 1) / nk
    for a in ks:
        for b in ks:
            H2, Ci, Vi = _blocks(phi, a, b)
            cv = max(cv, np.max(np.abs(H2[np.ix_(Ci, Vi)])))
            S = c_operator(phi, a, b)
            # Phi = 1: S is a folded scalar symbol, eigenvalues of the two
            # momenta k1/2 and k1/2 + pi on the C supercell
            want = sorted(2 * np.cos(t) + 2 * np.cos(b) - 2 * np.cos(t - b)
                          for t in (a / 2, a / 2 + pi))
            sym = max(sym, np.max(np.abs(np.linalg.eigvalsh(S) - want)))
            E = np.linalg.eigvalsh(C.magnetic_bloch(phi, a, b))
            allE += list(E)
            CVi = Ci + Vi
            ev_split = np.sort(np.concatenate(
                [np.linalg.eigvalsh(H2[np.ix_(Ci, Ci)]),
                 np.linalg.eigvalsh(H2[np.ix_(Vi, Vi)])]))
            split = max(split, np.max(np.abs(
                np.sort(np.linalg.eigvalsh(H2[np.ix_(CVi, CVi)])) - ev_split)))
    allE = np.sort(allE)
    return {"cv_block": cv, "S_symbol": sym, "split": split,
            "range": (allE[0], allE[-1]),
            "largest_gap": float(np.max(np.diff(allE))),
            "gaps": C.gap_cherns(phi)}


# ---------------------------------------------------------------------------
# 5. the Chern obstruction
# ---------------------------------------------------------------------------

def _v_hexagon_flux(phi):
    pts = list(C._V)
    tot = 0.0
    for i in range(6):
        tot += C.landau_phase(pts[i], pts[(i + 1) % 6], phi)
    return tot / (2 * pi)


def chern_obstruction(phis=(Fr(1, 6), Fr(1, 3)), control=Fr(1, 10)):
    r"""At integer flux per cell, [S, U] = 0 and the V hexagon flux is an
    integer, while the substrate has gaps of Chern +-1."""
    rng = np.random.default_rng(3)
    out = {}
    for phi in list(phis) + [control]:
        comm = 0.0
        for _ in range(3):
            k1, k2 = rng.uniform(0, 2 * pi, 2)
            S = c_operator(phi, k1, k2, SHORT)
            U = c_operator(phi, k1, k2, LONG)
            comm = max(comm, np.max(np.abs(S @ U - U @ S)))
        out[phi] = {"Phi": 6 * phi, "commutator": comm,
                    "v_hexagon_flux": _v_hexagon_flux(phi)}
        if phi != control:
            out[phi]["gap_cherns"] = [g["chern_int"] for g in C.gap_cherns(phi)]
    return out


# ---------------------------------------------------------------------------
# 6. control: the naive quantisation of the zero-field charpoly
# ---------------------------------------------------------------------------

def naive_quantisation(phi, ntrial=3, seed=0):
    r"""Max relative |det(E - H) - det P(E, S, U)| over random E and twists,
    P(E, s, u) = E^6 - (12 + s) E^4 - 3 (s - 6) E^2 - (12 - s^2 + 4 u)."""
    phi = Fr(phi)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(ntrial):
        k1, k2 = rng.uniform(0, 2 * pi, 2)
        cp = np.poly(C.magnetic_bloch(phi, k1, k2))
        S = c_operator(phi, k1, k2, SHORT)
        U = c_operator(phi, k1, k2, LONG)
        I = np.eye(S.shape[0])
        for E in rng.normal(size=6) * 2:
            O = E ** 6 * I - E ** 4 * (12 * I + S) - 3 * E ** 2 * (S - 6 * I) \
                - (12 * I - S @ S + 4 * U)
            lhs = np.polyval(cp, E)
            worst = max(worst, abs(lhs - np.linalg.det(O)) / abs(lhs))
    return worst


# ---------------------------------------------------------------------------

def status_mirror():
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    zf = zero_field_invariants()
    ok("Delta Delta-bar = 12 - s^2 + 4u, u the long-root hexagon",
       zf["DeltaDeltabar_is_12_minus_s2_plus_4u"] and zf["charpoly_in_s_u"])
    ok("u is the hexagon on the index-3 sublattice; s, u independent",
       zf["long_root_index"] == 3 and _long_is_image()
       and abs(zf["jacobian_s_u"]) > 1e-3)
    good = True
    for phi in (Fr(0), Fr(1, 2), Fr(1, 3), Fr(1, 4), Fr(1, 6), Fr(2, 7)):
        si = square_identity(phi)
        good &= max(si["residual"].values()) < 1e-12
        good &= si["diag"] == {"C": [6.0], "E": [4.0], "V": [3.0]}
    ok("H^2 = [[6 + S, [2]_q K], [., 3 + H_V]] on C u V (patch)", good)
    ok("H^2_CC = 6 + S on the magnetic supercell", supercell_identity() < 1e-12)
    ok("S = quantised B_3 hexagon at Phi = 6 phi mod 2 (moments n <= 10)",
       all(hexagon_match(p)["rel_diff"] < 1e-9
           for p in (Fr(1, 12), Fr(1, 18), Fr(1, 10), Fr(1, 6), Fr(2, 7))))
    hf = half_flux()
    ok("phi = 1/2: C-V decouple, S pi-flux triangular, gapless on [-3, 3]",
       hf["cv_block"] < 1e-12 and hf["S_symbol"] < 1e-12
       and hf["split"] < 1e-12 and abs(hf["range"][0] + 3) < 0.05
       and abs(hf["range"][1] - 3) < 0.05 and hf["gaps"] == [])
    co = chern_obstruction()
    ok("integer Phi: [S, U] = 0 and V flux integer, yet gap Chern +-1",
       all(co[p]["commutator"] < 1e-12
           and abs(co[p]["v_hexagon_flux"] - round(co[p]["v_hexagon_flux"])) < 1e-12
           and any(abs(c) == 1 for c in co[p]["gap_cherns"])
           for p in (Fr(1, 6), Fr(1, 3)))
       and co[Fr(1, 10)]["commutator"] > 1e-2)
    ok("naive Weyl charpoly: exact at zero field, fails in a field",
       naive_quantisation(Fr(1)) < 1e-9
       and all(naive_quantisation(p) > 1e3 for p in (Fr(1, 6), Fr(1, 12), Fr(1, 9))))
    for name, c in checks:
        print("  {:<66} {}".format(name, "ok" if c else "FAIL"))
    return all(c for _, c in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if status_mirror() else 1)
