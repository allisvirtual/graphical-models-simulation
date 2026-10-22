"""Shared helpers for the three Part A simulations."""

import numpy as np

rng = np.random.default_rng(0)

N = 200_000


def bernoulli(p, n=N):
    """Draw n 0/1 values. p is one probability, or one probability per case."""
    return (rng.random(n) < p).astype(int)


def dependence(a, c, b, names):
    """Print the relationship between A and C on all data, then within each level of B."""
    A, C, B = names

    def line(label, a_rows, c_rows):
        hi = c_rows[a_rows == 1].mean()
        lo = c_rows[a_rows == 0].mean()
        corr = np.corrcoef(a_rows, c_rows)[0, 1]
        print(f"  {label:<16} P({C}=1|{A}=1)={hi:.3f}   P({C}=1|{A}=0)={lo:.3f}"
              f"   gap={hi - lo:+.3f}   corr={corr:+.3f}")

    line("all data", a, c)
    for v in (0, 1):
        mask = b == v
        line(f"only {B}={v}", a[mask], c[mask])
    print()
