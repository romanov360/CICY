r"""
pyCICY.cusp_lll -- the cusp substrate's lowest Landau level at one flux
quantum, and theta_1's odd characteristic as its zero at the C site
(Part XIII).

Part XI identified F = e(beta/12) theta_1(tau, mu)/eta(tau) as the function
whose multiplier is the holonomy character, with its zero on mu in Z + tau Z:
the C site of Part X's substrate. In the continuum, theta_1 is the lowest
Landau level (LLL) on a torus with one flux quantum, and its oddness -- the
characteristic (1/2, 1/2) -- is a boundary condition. This module asks the
lattice the same question. Put one flux quantum through each cusp domain
(Phi = 6 phi = 1, kite flux phi = 1/6) and look at the substrate's lowest
band.

1. **One flux quantum needs no supercell** (:func:`magnetic_bloch`). With
   A = 2 pi Phi u dv and Phi an integer, magnetic translations by the cusp
   lattice commute. The covariant boundary conditions are

       psi(u + 1, v) = e^{i k1} e^{2 pi i Phi v} psi(u, v),
       psi(u, v + 1) = e^{i k2} psi(u, v),

   so the magnetic Bloch Hamiltonian is 6 x 6, one orbital per substrate site,
   with the same Landau gauge as :func:`pyCICY.cusp_hofstadter.landau_phase`.
   The sign is -A (hopping -1), the lattice analogue of -(nabla - iA)^2; the
   graph is bipartite, so +A has the same states up to a sublattice sign.
2. **The lowest band is a Landau level** (:func:`lowest_band`). It is
   isolated, fills [-2 sqrt 3, -sqrt 6] (the endpoints from the
   characteristic polynomials at the symmetric twists), and has Chern number
   1. That is two routes: FHS on this 1-cell bundle, and
   :func:`pyCICY.cusp_hofstadter.gap_cherns` on the 6 x 1 supercell at
   phi = 1/6, which knows nothing of the 1-cell convention.
3. **Its zeros are theta_1's** (:func:`site_zeros`). A Chern-1 band has no
   global nonvanishing section, so each site amplitude psi_s(k) must vanish
   somewhere on the twist torus. It vanishes exactly once, at

       k / 2 pi = (1/2 - v_s, 1/2 + u_s)   mod 1,

   for each of the six sites (u_s, v_s). In particular the C site (0, 0) is
   reached at the antiperiodic twist (1/2, 1/2): the odd spin structure. The
   E site (1/2, 1/2) is reached at the periodic twist (0, 0).
4. **The same law in the continuum** (:func:`continuum_twist`). The LLL of
   -(nabla - iA)^2 on the cusp torus (tau = e^{i pi / 3}, same gauge, same
   boundary conditions) is e^{g} e^{a z + b z^2} theta_1((z - z0)/omega_1 | tau)
   with g fixed by the gauge. Fitting a, b to the boundary conditions for a
   prescribed zero z0 and reading off the twist gives the same affine law, at
   the six sites and at generic points. So the lattice LLL and the continuum
   LLL select the same characteristic. Lattice and continuum states agree to
   overlap > 0.97 on the six sites (:func:`overlaps`).
5. **Controls** (:func:`controls`). With no field the lowest band has Chern
   number 0. At the periodic twist the C amplitude is not zero; the zero
   is at the E site (1/2, 1/2), the even characteristic.

What this is: a lattice check that the oddness of theta_1, which Part IX
needed for algebraic reasons (the odd theta characteristic in chi), is the
boundary condition at which the substrate's lowest Landau level vanishes on
the C site. What it is not: a derivation of chi from the electron model, or
any claim that electrons live on the six-sphere. The zero positions are
measured to 1e-15 at the predicted twists, not proved; the characteristic
polynomials at the symmetric twists have integer coefficients to 1e-13 and
are rounded and factored over Q (:func:`band_ends_exact`), which gives the
band endpoints -sqrt 6 (C-zero twist) and -2 sqrt 3 (V-zero twist).
"""

from fractions import Fraction as Fr
from math import pi, sqrt

import numpy as np
from sympy import I as _I, Matrix as _Matrix, exp as _exp, \
    factor as _factor, nsimplify as _nsimplify, pi as _pi, symbols as _symbols

from . import cusp_hofstadter as C

__all__ = [
    "magnetic_bloch", "magnetic_bloch_exact", "lowest_band", "band_ends_exact",
    "predicted_twist", "site_zeros", "continuum_twist", "continuum_state",
    "overlaps", "controls", "status_lll",
]

_S = [(float(a), float(b)) for a, b in C.SITES]
_N = len(_S)


def _line(r1, r2, Phi):
    """int_{r1}^{r2} A along the straight segment, A = 2 pi Phi u dv."""
    return 2 * pi * Phi * (r1[0] + r2[0]) / 2 * (r2[1] - r1[1])


def magnetic_bloch(k1, k2, Phi=1, sign=-1):
    r"""The 6 x 6 magnetic Bloch Hamiltonian for Phi flux quanta per domain.

    H_{a,b} = sign * e^{i int_{r_b + R}^{r_a} A} * (covariant phase of
    psi(r_b + R)), so H is gauge-covariant in the convention of the module
    docstring. Phi must be an integer.
    """
    H = np.zeros((_N, _N), complex)
    for a, b, R in C.BONDS:
        ra = _S[a]
        rb = (_S[b][0] + R[0], _S[b][1] + R[1])
        amp = sign * np.exp(-1j * _line(ra, rb, Phi)) \
            * np.exp(1j * (k1 * R[0] + k2 * R[1])) \
            * np.exp(2j * pi * Phi * R[0] * _S[b][1])
        H[a, b] += amp
        H[b, a] += np.conj(amp)
    return H


def magnetic_bloch_exact(t1, t2, Phi=1, sign=-1):
    """:func:`magnetic_bloch` at twist k = 2 pi (t1, t2), t rational, in sympy."""
    H = _Matrix.zeros(_N, _N)
    S = C.SITES
    for a, b, R in C.BONDS:
        ra = S[a]
        rb = (S[b][0] + R[0], S[b][1] + R[1])
        turns = -Phi * (ra[0] + rb[0]) / 2 * (rb[1] - ra[1]) \
            + Fr(t1) * R[0] + Fr(t2) * R[1] + Phi * R[0] * S[b][1]
        amp = sign * _exp(2 * _pi * _I * _nsimplify(turns))
        H[a, b] += amp
        H[b, a] += amp.conjugate()
    return H


def _low(k1, k2, Phi=1):
    return np.linalg.eigh(magnetic_bloch(k1, k2, Phi))[1][:, 0]


def _fhs(Phi=1, band=0, nk=30):
    ks = 2 * pi * np.arange(nk) / nk
    U = [[np.linalg.eigh(magnetic_bloch(a, b, Phi))[1][:, band] for b in ks]
         for a in ks]
    F = 0.0
    for i in range(nk):
        for j in range(nk):
            a, b = U[i][j], U[(i + 1) % nk][j]
            c, d = U[(i + 1) % nk][(j + 1) % nk], U[i][(j + 1) % nk]
            F += np.angle(np.vdot(a, b) * np.vdot(b, c) * np.vdot(c, d)
                          * np.vdot(d, a))
    return F / (2 * pi)


def lowest_band(nk=30):
    """Range of the lowest band, the gap above it, and its Chern number by
    two routes (1-cell FHS here; the 6 x 1 supercell of Part X)."""
    ks = 2 * pi * np.arange(nk) / nk
    E = np.array([np.linalg.eigvalsh(magnetic_bloch(a, b))
                  for a in ks for b in ks])
    sup = C.gap_cherns(Fr(1, 6), nk=18, nkgap=18)
    # Part X uses hopping +1, whose top band is this (-1) model's lowest band
    top = [g for g in sup if g["n_below"] == 30]
    return {"range": (float(E[:, 0].min()), float(E[:, 0].max())),
            "next_band_min": float(E[:, 1].min()),
            "chern_1cell": _fhs(nk=nk),
            "chern_supercell_top_of_plus_A": top[0]["chern"] if top else None}


def band_ends_exact():
    """Characteristic polynomials at the twists where the C, E and V sites
    are zeros, and the lowest root of each.

    The entries are 12th roots of unity, so the coefficients are algebraic
    integers; here they come out integers to 1e-12 in floating point, are
    rounded, and factored over Q (a symbolic determinant is minutes slow)."""
    E = _symbols("E")
    out = {}
    for name, t in (("C", (Fr(1, 2), Fr(1, 2))), ("E", (0, 0)),
                    ("V", (Fr(1, 6), Fr(5, 6)))):
        c = np.poly(magnetic_bloch(2 * pi * float(t[0]), 2 * pi * float(t[1])))
        ci = np.round(c.real).astype(int)
        err = float(np.abs(c - ci).max())
        poly = sum(int(x) * E ** (_N - i) for i, x in enumerate(ci))
        out[name] = {"twist": t, "charpoly": _factor(poly), "round_err": err,
                     "lowest": float(np.linalg.eigvalsh(
                         magnetic_bloch(2 * pi * float(t[0]),
                                        2 * pi * float(t[1])))[0])}
    return out


def predicted_twist(u, v):
    """The twist (in turns) at which the LLL vanishes at (u, v)."""
    return ((Fr(1, 2) - Fr(v)) % 1, (Fr(1, 2) + Fr(u)) % 1)


def site_zeros(nk=60):
    """For each site: |psi_s| at the predicted twist, and the minimum of
    |psi_s|^2 on the grid away from it (the zero is unique)."""
    ks = np.arange(nk) / nk
    P = np.array([[np.abs(_low(2 * pi * a, 2 * pi * b)) ** 2 for b in ks]
                  for a in ks])
    out = []
    for s, (u, v) in enumerate(C.SITES):
        t = predicted_twist(u, v)
        at = abs(_low(2 * pi * float(t[0]), 2 * pi * float(t[1]))[s])
        d1 = np.minimum(np.abs(ks[:, None] - float(t[0])),
                        1 - np.abs(ks[:, None] - float(t[0])))
        d2 = np.minimum(np.abs(ks[None, :] - float(t[1])),
                        1 - np.abs(ks[None, :] - float(t[1])))
        far = (d1 ** 2 + d2 ** 2) > 0.1 ** 2
        out.append({"site": C.SITE_KIND[s], "uv": (u, v), "twist": t,
                    "abs_at_twist": float(at),
                    "min_away": float(P[..., s][far].min())})
    return out


# ---------------------------------------------------------------------------
# the continuum LLL on the same torus, same gauge
# ---------------------------------------------------------------------------

_M = np.column_stack([C.E1, C.E2])
_Mi = np.linalg.inv(_M)
_P = (_Mi[0, 0] - 1j * _Mi[0, 1]) / 2          # u = P z + conj(P) conj(z)
_CV = _Mi[1, 0] + 1j * _Mi[1, 1]               # dv = Re(conj(CV)... ) as A_x + i A_y
_W1 = complex(*C.E1)
_W2 = complex(*C.E2)
TAU = _W2 / _W1


def _logpsi(z, z0, a, b):
    import mpmath as mp
    q = mp.exp(1j * mp.pi * TAU)
    g = 1j * pi * _CV * (_P * z * np.conj(z) + np.conj(_P) * np.conj(z) ** 2 / 2)
    th = complex(mp.jtheta(1, mp.pi * (z - z0) / _W1, q))
    return g + a * z + b * z * z + np.log(th)


def _fit(z0, seed=1):
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    pts = [complex(*(_M @ rng.uniform(0.1, 0.9, 2))) for _ in range(10)]

    def resid(x):
        a, b = x[0] + 1j * x[1], x[2] + 1j * x[3]
        r = []
        for w, extra in ((_W1, True), (_W2, False)):
            L = [_logpsi(z + w, z0, a, b) - _logpsi(z, z0, a, b)
                 - (2j * pi * (_Mi[1] @ [z.real, z.imag]) if extra else 0)
                 for z in pts]
            r += [l.real for l in L]
            ph = np.exp(1j * np.array([l.imag for l in L]))
            r += list(np.abs(ph - ph[0]))
        return r

    best = None
    for _ in range(20):
        s = least_squares(resid, rng.normal(size=4), xtol=1e-15, ftol=1e-15,
                          gtol=1e-15)
        if best is None or s.cost < best.cost:
            best = s
        if s.cost < 1e-20:
            break
    return best.x[0] + 1j * best.x[1], best.x[2] + 1j * best.x[3], best.cost


def continuum_twist(u0, v0):
    """Twist (in turns) of the continuum LLL whose zero is at (u0, v0), and
    the residual of the boundary-condition fit."""
    z0 = complex(*(_M @ np.array([float(u0), float(v0)])))
    a, b, cost = _fit(z0)
    z = 0.31 * _W1 + 0.47 * _W2
    k1 = (_logpsi(z + _W1, z0, a, b) - _logpsi(z, z0, a, b)
          - 2j * pi * (_Mi[1] @ [z.real, z.imag])).imag / (2 * pi) % 1
    k2 = (_logpsi(z + _W2, z0, a, b) - _logpsi(z, z0, a, b)).imag / (2 * pi) % 1
    return (float(k1), float(k2)), float(cost)


def continuum_state(u0, v0):
    """The continuum LLL with zero at (u0, v0), sampled on the six sites."""
    z0 = complex(*(_M @ np.array([float(u0), float(v0)])))
    a, b, _ = _fit(z0)
    vals = []
    for u, v in _S:
        z = complex(*(_M @ np.array([u, v])))
        vals.append(0 if abs(z - z0) < 1e-12 else np.exp(_logpsi(z, z0, a, b)))
    vals = np.array(vals)
    return vals / np.linalg.norm(vals)


def overlaps(points=((0, 0), (0.2, 0.7), (0.5, 0.5), (Fr(1, 3), Fr(1, 3)),
                     (0.37, 0.11))):
    """|<lattice LLL, continuum LLL>| on the six sites, at the twist that puts
    the zero at each point."""
    out = []
    for u0, v0 in points:
        t = predicted_twist(Fr(u0).limit_denominator(1000),
                            Fr(v0).limit_denominator(1000))
        lat = _low(2 * pi * float(t[0]), 2 * pi * float(t[1]))
        out.append(((u0, v0), float(abs(np.vdot(lat, continuum_state(u0, v0))))))
    return out


def controls():
    """No field: Chern 0. Periodic twist: C amplitude nonzero, zero on the
    E site (1/2, 1/2)."""
    psi = _low(0.0, 0.0)
    iE = C.SITES.index((Fr(1, 2), Fr(1, 2)))
    return {"chern_no_field": _fhs(Phi=0, nk=20),
            "abs_C_at_periodic": float(abs(psi[0])),
            "abs_E_half_half_at_periodic": float(abs(psi[iE]))}


def status_lll():
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    lb = lowest_band(24)
    ok("lowest band isolated in [-2 sqrt 3, -sqrt 6], next at -sqrt 3",
       abs(lb["range"][0] + 2 * sqrt(3)) < 1e-9
       and abs(lb["range"][1] + sqrt(6)) < 1e-9
       and abs(lb["next_band_min"] + sqrt(3)) < 1e-9)
    ok("Chern 1 by two routes (1-cell FHS, 6 x 1 supercell)",
       abs(lb["chern_1cell"] - 1) < 1e-6
       and lb["chern_supercell_top_of_plus_A"] is not None
       and abs(lb["chern_supercell_top_of_plus_A"] - 1) < 1e-6)
    be = band_ends_exact()
    ok("band ends: top at the C-zero twist, bottom at the V-zero twist",
       str(be["C"]["charpoly"]) == "E**4*(E**2 - 6)"
       and str(be["V"]["charpoly"]) == "E**2*(E**2 - 12)*(E**2 - 3)"
       and abs(be["C"]["lowest"] + sqrt(6)) < 1e-12
       and abs(be["V"]["lowest"] + 2 * sqrt(3)) < 1e-12)
    sz = site_zeros(48)
    ok("each site amplitude vanishes at k = (1/2 - v, 1/2 + u)",
       all(z["abs_at_twist"] < 1e-12 for z in sz))
    ok("and nowhere else on the twist torus",
       all(z["min_away"] > 1e-4 for z in sz))
    ok("the C site at the odd twist (1/2, 1/2)",
       sz[0]["site"] == "C" and sz[0]["twist"] == (Fr(1, 2), Fr(1, 2)))
    good = True
    for u, v in list(C.SITES) + [(0.2, 0.7)]:
        (k1, k2), cost = continuum_twist(u, v)
        t = predicted_twist(Fr(u).limit_denominator(100),
                            Fr(v).limit_denominator(100))
        d = [min(abs(x - float(y)), 1 - abs(x - float(y)))
             for x, y in zip((k1, k2), t)]
        good &= cost < 1e-16 and max(d) < 1e-6
    ok("continuum theta_1 LLL: the same twist law", good)
    ok("lattice-continuum overlap > 0.97",
       all(o > 0.97 for _, o in overlaps()))
    ct = controls()
    ok("controls: no field Chern 0; periodic twist zero on E, not C",
       abs(ct["chern_no_field"]) < 1e-6 and ct["abs_C_at_periodic"] > 1e-2
       and ct["abs_E_half_half_at_periodic"] < 1e-12)
    for name, c in checks:
        print("  {:<62} {}".format(name, "ok" if c else "FAIL"))
    return all(c for _, c in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if status_lll() else 1)
