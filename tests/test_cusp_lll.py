"""
Tests for pyCICY.cusp_lll (Part XIII: the cusp substrate's lowest Landau level
at one flux quantum, and theta_1's odd characteristic as its C-site zero).

  [1] bloch        6 x 6 magnetic Bloch Hamiltonian is Hermitian; exact and
                   float matrices agree
  [2] band         lowest band isolated in [-2 sqrt 3, -sqrt 6]; Chern 1 by
                   two routes; band ends at the C- and V-zero twists
  [3] zeros        each site amplitude vanishes once, at (1/2 - v, 1/2 + u);
                   the C site at the odd twist (1/2, 1/2)
  [4] continuum    the theta_1 LLL obeys the same twist law; overlaps > 0.97
  [5] controls     no field: Chern 0; periodic twist: zero on E, not C

Run with:  python3 tests/test_cusp_lll.py   or   python3 run_tests.py
"""

import os
import sys
from fractions import Fraction as Fr
from math import pi, sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pyCICY import cusp_lll as L

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>12} {}".format(name, str(got)[:12],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def test_bloch():
    print("[1] magnetic Bloch Hamiltonian")
    for t in ((0, 0), (Fr(1, 2), Fr(1, 2)), (Fr(1, 6), Fr(5, 6)), (Fr(1, 3), Fr(1, 4))):
        H = L.magnetic_bloch(2 * pi * float(t[0]), 2 * pi * float(t[1]))
        Hx = np.array(L.magnetic_bloch_exact(*t).evalf(), dtype=complex)
        check_true("t = %s: Hermitian, exact = float" % (t,),
                   np.allclose(H, H.conj().T) and np.allclose(H, Hx))


def test_band():
    print("[2] lowest band")
    lb = L.lowest_band(24)
    check_true("range [-2 sqrt 3, -sqrt 6]",
               abs(lb["range"][0] + 2 * sqrt(3)) < 1e-9
               and abs(lb["range"][1] + sqrt(6)) < 1e-9)
    check_true("next band starts at -sqrt 3", abs(lb["next_band_min"] + sqrt(3)) < 1e-9)
    check("Chern, 1-cell FHS", round(lb["chern_1cell"], 6), 1)
    check("Chern, 6 x 1 supercell (Part X)",
          round(lb["chern_supercell_top_of_plus_A"], 6), 1)
    be = L.band_ends_exact()
    check("charpoly at C-zero twist", str(be["C"]["charpoly"]), "E**4*(E**2 - 6)")
    check("charpoly at E-zero twist", str(be["E"]["charpoly"]),
          "(E**2 - 2)*(E**4 - 12*E**2 + 8)")
    check("charpoly at V-zero twist", str(be["V"]["charpoly"]),
          "E**2*(E**2 - 12)*(E**2 - 3)")
    check_true("coefficients integer to 1e-12",
               all(v["round_err"] < 1e-12 for v in be.values()))


def test_zeros():
    print("[3] site zeros")
    sz = L.site_zeros(48)
    for z in sz:
        check_true("%s %s: zero at %s, unique" % (z["site"], z["uv"], z["twist"]),
                   z["abs_at_twist"] < 1e-12 and z["min_away"] > 1e-4)
    check("C site twist", sz[0]["twist"], (Fr(1, 2), Fr(1, 2)))


def test_continuum():
    print("[4] continuum theta_1 LLL")
    for u, v in list(L.C.SITES) + [(0.2, 0.7)]:
        (k1, k2), cost = L.continuum_twist(u, v)
        t = L.predicted_twist(Fr(u).limit_denominator(100),
                              Fr(v).limit_denominator(100))
        d = max(min(abs(x - float(y)), 1 - abs(x - float(y)))
                for x, y in zip((k1, k2), t))
        check_true("(%s, %s): twist law, fit residual" % (u, v),
                   d < 1e-6 and cost < 1e-16)
    for p, o in L.overlaps():
        check_true("overlap at %s > 0.97" % (p,), o > 0.97)


def test_controls():
    print("[5] controls")
    ct = L.controls()
    check("no field: Chern", round(ct["chern_no_field"], 6), 0)
    check_true("periodic twist: C amplitude nonzero", ct["abs_C_at_periodic"] > 1e-2)
    check_true("periodic twist: E(1/2, 1/2) amplitude zero",
               ct["abs_E_half_half_at_periodic"] < 1e-12)


def main():
    for section in (test_bloch, test_band, test_zeros, test_continuum,
                    test_controls):
        try:
            section()
        except Exception as exc:
            print("  {:<58} {:>12} FAIL".format(section.__name__ + " raised",
                                                type(exc).__name__))
            print("      " + str(exc)[:300])
            FAILURES.append("%s raised %s" % (section.__name__,
                                              type(exc).__name__))
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all cusp_lll tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
