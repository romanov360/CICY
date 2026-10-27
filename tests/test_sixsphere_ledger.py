"""
Tests for pyCICY.sixsphere_ledger (the generated evidence ledger, Parts IX-XIV).

  [1] coverage     the table and the printed checks match one to one; all pass
  [2] vocabulary   Q rows name a source, X rows a route; controls are labelled
  [3] Theorem XI   the quoted inputs on its critical path are exactly two
  [4] document     the generated block in AlpogeSixSphere.md is current
  [5] controls     a dropped row, an extra row and a stray Q each fail the selftest

Run with:  python3 tests/test_sixsphere_ledger.py   or   python3 run_tests.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyCICY import sixsphere_ledger as G

FAILURES = []
ROWS = []


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


def test_coverage():
    print("[1] coverage")
    ROWS.extend(G.collect())
    st = G.selftest(ROWS)
    for k in ("every check passes", "every printed check has a row",
              "every row has a printed check", "check names unique"):
        check_true(k, st[k])
    check("checks from six status functions",
          len({r["source"] for r in ROWS}), 6)
    check("rows", G.counts()["rows"], len(G.KIND))


def test_vocabulary():
    print("[2] vocabulary")
    st = G.selftest(ROWS)
    for k in ("kinds in the vocabulary", "every Q has a source, every source a Q",
              "every X has a route, every route an X", "controls named as controls",
              "L only where a Lean fact is named", "no Q or P row linked to Lean"):
        check_true(k, st[k])
    check("no Lean fact from Part IX on (lead 5)", G.counts()["total"]["L"], 0)


def test_theorem_xi():
    print("[3] Theorem XI")
    cq = G.critical_quoted()
    check("quoted inputs on the critical path", len(cq), 2)
    check_true("theta laws (Mumford) among them", any("Mumford" in q for q in cq))
    check_true("SL(2, Z) generators among them", any("SL(2, Z)" in q for q in cq))
    check_true("every critical row is derived",
               all("D" in G.KIND[c] for c in G.CRITICAL))


def test_document():
    print("[4] document")
    text = open(os.path.join(ROOT, "AlpogeSixSphere.md"), encoding="utf-8").read()
    i, j = text.index(G.BEGIN), text.index(G.END) + len(G.END)
    check_true("generated block is current", text[i:j] == G.markdown())


def test_controls():
    print("[5] controls")
    key = next(iter(G.KIND))
    saved = G.KIND.pop(key)
    try:
        check_true("dropped row fails", not G.selftest(ROWS)["every printed check has a row"])
    finally:
        G.KIND[key] = saved
    extra = ("mirror", "a check nobody prints")
    G.KIND[extra] = "D"
    try:
        check_true("extra row fails", not G.selftest(ROWS)["every row has a printed check"])
    finally:
        del G.KIND[extra]
    G.KIND[key] = saved + "Q"
    try:
        check_true("unsourced Q fails",
                   not G.selftest(ROWS)["every Q has a source, every source a Q"])
    finally:
        G.KIND[key] = saved
    check_true("restored table passes", all(G.selftest(ROWS).values()))


def main():
    for section in (test_coverage, test_vocabulary, test_theorem_xi,
                    test_document, test_controls):
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
    print("all sixsphere_ledger tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
