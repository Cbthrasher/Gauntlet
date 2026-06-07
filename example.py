"""example.py — a 60-second tour of the Gauntlet.

Three synthetic strategies (fixed seed, so you get the same numbers every run):

  1. A real, robust edge, declared honestly                 -> SURVIVES
  2. The SAME-looking edge, but it was the best of 300 tries -> KILLED
     (the Deflated Sharpe is the only gate that catches this, and it's the
      one almost nobody runs)
  3. A razor-thin edge that ordinary trading costs erase     -> KILLED

Run:  python example.py
"""
import random

import gauntlet as g

SEED = 20260606


def make_returns(mean, sd, n, seed):
    """A flat list of per-trade returns (pips). Your real returns go here instead."""
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


# --------------------------------------------------------------------------- #
# 1) A REAL edge — one honest strategy, no variant-hunting. Should SURVIVE.
# --------------------------------------------------------------------------- #
print("\n" + "#" * 70)
print("# 1) A REAL edge — one honest strategy, run through the whole gauntlet")
print("#" * 70)
real = make_returns(mean=3.0, sd=9.0, n=600, seed=SEED)
regimes = ["CALM" if i % 2 else "WILD" for i in range(len(real))]
g.run_gates(
    real,
    n_trials=1,  # you tried exactly one thing and it worked
    mechanism="(demo) you can name who is on the other side and why they keep losing",
    regimes=regimes,
    extra_cost_pips=0.5,
    label="real_edge",
)

# --------------------------------------------------------------------------- #
# 2) The DATA-SNOOPING TRAP. Same trades, two different honesty levels.
# --------------------------------------------------------------------------- #
print("\n" + "#" * 70)
print("# 2) The DATA-SNOOPING TRAP — a thin edge that was the best of 300 tries")
print("#" * 70)
snooped = make_returns(mean=1.2, sd=10.0, n=500, seed=SEED + 1)
honest_1 = g.deflated_sharpe(snooped, n_trials=1)
honest_300 = g.deflated_sharpe(snooped, n_trials=300)
print(f"  Per-trade Sharpe of the strategy you kept:  {honest_1['sr']:+.3f}")
print(f"\n  If you pretend it is the only thing you tried (n_trials=1):")
print(f"      Deflated Sharpe = {honest_1['dsr']:.3f}  ->  "
      f"{'PASS' if honest_1['pass'] else 'FAIL'}")
print(f"  If you admit you tried 300 variants and kept the best (n_trials=300):")
print(f"      Deflated Sharpe = {honest_300['dsr']:.3f}  ->  "
      f"{'PASS' if honest_300['pass'] else 'FAIL'}")
print(f"\n  Same trades. The only thing that changed is honesty about how hard you")
print(f"  searched. That gap is exactly how a backtest fools you — G3a is the gate")
print(f"  that makes the gap visible.")

# --------------------------------------------------------------------------- #
# 3) BONUS — a razor-thin edge that ordinary costs erase. Should be KILLED.
# --------------------------------------------------------------------------- #
print("\n" + "#" * 70)
print("# 3) BONUS — a razor-thin edge that trading costs flip negative")
print("#" * 70)
thin = make_returns(mean=0.5, sd=2.0, n=600, seed=SEED + 2)
cs = g.cost_stress(thin, extra_cost_pips=0.6)
print(f"  mean per trade before costs: {cs['base_mean']:+.2f}p")
print(f"  mean per trade after +0.6p:  {cs['stressed_mean']:+.2f}p")
print(f"  expectancy retained: {cs['retention'] * 100:.0f}%   inverts sign: {cs['inverts']}"
      f"   ->  {'PASS' if cs['pass'] else 'FAIL'}")
print(f"  A real edge degrades gracefully under friction. A fake one flips sign.\n")
