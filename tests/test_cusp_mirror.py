"""
Tests for pyCICY.cusp_mirror (Part XIV: the cusp substrate in a field against
the quantised dP_6 hexagon).

  [1] zero field   Delta Delta-bar = 12 - s^2 + 4u; u the long-root hexagon on
                   the index-3 sublattice; s, u independent
  [2] identity     H^2 on C u V = [[6 + S, [2]_q K], [., 3 + H_V]] on a patch,
                   for several fluxes; H^2_CC = 6 + S on the supercell
  [3] hexagon      S = B_3 quantum curve at Phi = 6 phi mod 2 (moments)
  [4] half flux    phi = 1/2: C and V decouple, gapless on [-3, 3]
  [5] obstruction  integer Phi: [S, U] = 0, V flux integer, gap Chern +-1
  [6] control      naive Weyl charpoly exact at zero field, wrong in a field

Run with:  python3 tests/test_cusp_mirror.py   or   python3 run_tests.py
"""

import os
import sys
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import cusp_mirror as M

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


def test_zero_field():
    print("[1] zero field")
    zf = M.zero_field_invariants()
    check_true("Delta Delta-bar = 12 - s^2 + 4u",
               zf["DeltaDeltabar_is_12_minus_s2_plus_4u"])
    check_true("charpoly is a polynomial in (E, s, u)", zf["charpoly_in_s_u"])
    check("long roots span an index-3 sublattice", zf["long_root_index"], 3)
    check_true("long hexagon = short hexagon on that sublattice",
               zf["long_is_short_on_sublattice"])
    check_true("s, u independent (Jacobian nonzero)",
               abs(zf["jacobian_s_u"]) > 1e-3)


def test_identity():
    print("[2] H^2 identity")
    for phi in (Fr(0), Fr(1, 2), Fr(1, 3), Fr(1, 4), Fr(1, 5), Fr(1, 6),
                Fr(2, 7)):
        si = M.square_identity(phi)
        check_true("phi = %s: blocks to 1e-12, diag C 6, E 4, V 3" % phi,
                   max(si["residual"].values()) < 1e-12
                   and si["diag"] == {"C": [6.0], "E": [4.0], "V": [3.0]})
    check_true("supercell: H^2_CC = 6 + S", M.supercell_identity() < 1e-12)


def test_hexagon():
    print("[3] S is the quantised hexagon")
    for phi in (Fr(1, 12), Fr(1, 18), Fr(1, 10), Fr(1, 7), Fr(1, 6),
                Fr(2, 7), Fr(3, 11)):
        hm = M.hexagon_match(phi)
        check_true("phi = %s, Phi = %s: moments agree" % (phi, hm["Phi"]),
                   hm["rel_diff"] < 1e-9)
    check("Phi = 1 (phi = 1/6): pi-flux triangles, tr S^3 / dim",
          round(M.hexagon_match(Fr(1, 6))["m3"], 9), -12.0)


def test_half_flux():
    print("[4] phi = 1/2")
    hf = M.half_flux()
    check_true("C-V block of H^2 vanishes", hf["cv_block"] < 1e-12)
    check_true("S symbol 2cos k1 + 2cos k2 - 2cos(k1 - k2)",
               hf["S_symbol"] < 1e-12)
    check_true("spec H^2 = spec CC u spec VV", hf["split"] < 1e-12)
    check_true("spectrum reaches -3 and 3",
               abs(hf["range"][0] + 3) < 1e-9 and abs(hf["range"][1] - 3) < 1e-9)
    check("no gaps", hf["gaps"], [])


def test_obstruction():
    print("[5] Chern obstruction")
    co = M.chern_obstruction()
    for phi in (Fr(1, 6), Fr(1, 3)):
        c = co[phi]
        check_true("phi = %s: [S, U] = 0" % phi, c["commutator"] < 1e-12)
        check("phi = %s: V hexagon flux" % phi,
              round(c["v_hexagon_flux"], 9), float(6 * phi))
        check("phi = %s: substrate gap Chern numbers" % phi,
              c["gap_cherns"], [-1, 1])
    check_true("control phi = 1/10: [S, U] != 0",
               co[Fr(1, 10)]["commutator"] > 1e-2)


def test_control():
    print("[6] naive quantisation")
    check_true("zero field: exact", M.naive_quantisation(Fr(1)) < 1e-9)
    for phi in (Fr(1, 6), Fr(1, 12), Fr(1, 9)):
        check_true("phi = %s: fails" % phi, M.naive_quantisation(phi) > 1e3)


def main():
    for section in (test_zero_field, test_identity, test_hexagon,
                    test_half_flux, test_obstruction, test_control):
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
    print("all cusp_mirror tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
