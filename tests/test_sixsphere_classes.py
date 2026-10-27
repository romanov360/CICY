"""
Tests for pyCICY.sixsphere_classes (Part XII: the four near-miss classes are
the paper's monodromy group up to isogeny).

  [1] normal form   P_d^t Q P_d = d E03 + (6/d) E12, Pf 6, P_d unimodular
  [2] isogeny       D_d P_d^-1 T_i P_d D_d^-1 = (W1, W2) for every class;
                    h(0,1,0) conjugates (W1, W2) onto the paper's (T1, T2);
                    L_d = D_d Z^4 stable, index d^2; Q-conjugators det 1/d^2
  [3] index 12      Phi_d: Aut(Q_d, gamma)(Z) = G(Z); <W1, W2> = ker chi
  [4] exact p       reachable p mod 12 by periodicity: gcd d, least |p| = d
  [5] periods       class laws have the Z_d shape; F_d has multiplier chi o Phi_d
  [6] zeros         F_d = 0 on d(Z + tau Z), not at the d-torsion (the
                    conjecture refuted); characteristic control

Run with:  python3 tests/test_sixsphere_classes.py   or   python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational

from pyCICY import sixsphere_classes as C

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


def test_normal_form():
    print("\n[1] normal form")
    for d, v in C.adapted_forms().items():
        check("d = %d: divisibility of Q(gamma, .)" % d, v["d"], d)
        check("d = %d: Pfaffian" % d, v["pfaffian"], 6)
        check_true("d = %d: P^t Q P = d E03 + (6/d) E12, P fixes gamma" % d,
                   v["normal form"] and v["P fixes gamma"])


def test_isogeny():
    print("\n[2] isogeny")
    iso = C.isogeny()
    for d, v in iso["classes"].items():
        check_true("d = %d: image is (W1, W2)" % d, v["W = (W1, W2)"])
        check("d = %d: index of L_d" % d, v["index of L_d"], d * d)
        check_true("d = %d: L_d stable, U preserves Q_d" % d,
                   v["L_d stable"] and v["U preserve Q_d"])
    for k, v in iso["common"].items():
        check_true(k, v)
    for d, v in C.rational_conjugators().items():
        check_true("d = %d: C_d conjugates the paper's pair" % d, v["conjugates"])
        check("d = %d: |det C_d|" % d, abs(v["det"]), Rational(1, d * d))


def test_index():
    print("\n[3] structure group and index 12")
    for d, v in C.structure_isomorphism().items():
        check_true("d = %d: D^-1 G(Z) D integral (symbolic)" % d,
                   v["D^-1 G(Z) D integral"] and v["D^-t Q_d D^-1 = Q0/d"])
        check_true("d = %d: automorphism integral iff t in dZ^2 (box)" % d,
                   v["integral iff t in dZ^2"])
    ip = C.index_proof()
    check("chi(W1), chi(W2)", (ip["chi(W1)"], ip["chi(W2)"]), (0, 0))
    check_true("M1 M2 = T, M2 = S", ip["M1 M2 = T"] and ip["M2 = S"])
    check_true("<W1, W2> = ker chi", ip["proved"])


def test_exact_p():
    print("\n[4] reachable p, exactly")
    want = {1: [1, 5, 7, 11], 2: [2, 4, 8, 10], 3: [3, 9], 6: [0, 6]}
    for d in C.DS:
        rp = C.reachable_p(d)
        check("d = %d: residues mod 12" % d, rp["residues mod 12"], want[d])
        check("d = %d: gcd, least |p|" % d, (rp["gcd"], rp["least |p|"]), (d, d))
        check_true("d = %d: moduli absorbed by 12 l0" % d, rp["ma in 3Z, mb in 4Z"])


def test_periods():
    print("\n[5] periods and multiplier")
    for d, v in C.theta_check().items():
        for k, r in v["rows"].items():
            check_true("d = %d, %s: Z_d shape and multiplier" % (d, k),
                       r["shape_ok"] and r["ok"])


def test_zeros():
    print("\n[6] zeros: the origin orbit, not the d-torsion")
    for d in C.DS:
        z = C.zero_divisor_d(d)
        check_true("d = %d: F_d = 0 on d(Z + tau Z)" % d, z["max on d Lambda"] < 1e-15)
        if d > 1:
            check_true("d = %d: F_d != 0 at 1, tau, 1 + tau" % d,
                       z["min at 1, tau, 1+tau"] > 1e-3)
            cc = C.characteristic_control(d)
            check("d = %d: characteristic survivors of %d" % (d, cc["candidates"]),
                  cc["survivors"], [(0, 0)])
            check_true("d = %d: naive theta_1(tau, mu) not fixed" % d,
                       not cc["naive fixed"])


def main():
    for section in (test_normal_form, test_isogeny, test_index, test_exact_p,
                    test_periods, test_zeros):
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
    print("all sixsphere_classes tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
