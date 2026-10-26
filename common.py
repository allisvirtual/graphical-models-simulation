"""Shared helpers for parts A and B"""

import numpy as np

rng = np.random.default_rng(0)

N = 200_000


def bernoulli(p, n=N):
    """Draw n 0/1 values"""
    return (rng.random(n) < p).astype(int)


def show(label, a, c, A, C):
    hi = c[a == 1].mean()
    lo = c[a == 0].mean()
    corr = np.corrcoef(a, c)[0, 1]
    print(f"  {label:<16} P({C}=1|{A}=1)={hi:.3f}   P({C}=1|{A}=0)={lo:.3f}"
          f"   gap={hi - lo:+.3f}   corr={corr:+.3f}")


def dependence(a, c, b, names):
    A, C, B = names
    show("all data",      a,           c,           A, C)
    show(f"only {B}=0",   a[b == 0],   c[b == 0],   A, C)
    show(f"only {B}=1",   a[b == 1],   c[b == 1],   A, C)
    print()


def xtheny(a, b, names):
    """Print A's values conditioned on B"""
    A, B = names
    print(f"  {A} given {B}=1 OR 0")
    for v in (0, 1):
        mask = b == v
        mean = a[mask].mean()
        print(f"  {A} P({A}=1|{B}={v})={mean:.3f}")


def descendant(a, name):
    """Print the stats of descendant"""
    A = name
    mean = a.mean()
    print(f"  {name} P({A}=1)={mean:.3f}")