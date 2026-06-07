"""gauntlet.py — a cost-ordered KILL process for trading-strategy backtests.

The job of this module is to REJECT, not to bless. Give it a flat list of per-trade
returns (pips, dollars, R — whatever, one number per trade) and it runs a sequence of
statistical gates, cheapest-rejection-first, and tells you where the strategy dies. A
real edge survives friction and scrutiny; an overfit one collapses the moment either
one touches it. The whole point is to make it expensive to fool yourself.

Pure Python standard library — no numpy, scipy, or pandas. One file, drop it anywhere.

Gates (cheapest rejection first):
  G2  Sample floor    — >=400 trades AND >=2 distinct vol regimes, or it is too small
                        to tell skill from luck (cheap, kills fast).
  G3a Deflated Sharpe — adjusts the Sharpe for HOW MANY variants you tried (plus skew
                        and kurtosis) and returns P(edge is real). Cutoff 0.95. The
                        antidote to the classic self-deception: mine N variants, keep
                        the best, watch it forward-fail. (Bailey & Lopez de Prado.)
  G3b Block-bootstrap — resample RETURN BLOCKS (not single trades) so clustered losers
        max drawdown    keep their autocorrelation; sweep block length and read the
                        95th-pctile max drawdown to see the true tail, not a laundered one.
  G4a Cost stress     — add extra cost/slippage to every trade; require >=70% of
                        expectancy retained AND no sign flip. Real edges degrade
                        gracefully; fake ones invert the instant friction appears.
  G4b Regime split    — PF>1 in BOTH the calm and the stressed volatility halves (or,
                        for a declared specialist, across separated time blocks of its
                        own regime — see run_gates()).
  G5  Parameter plateau — sweep ONE knob; a real edge is a robust PLATEAU (the neighbors
                        of the best setting still work), an overfit one is a sharp PEAK
                        that collapses the moment you nudge the parameter.

G1 (economic mechanism — "name who is on the other side of this trade, and why they
keep losing") is deliberately NOT automatable; run_gates() prints a reminder so it is
never skipped silently. An automated PASS is necessary, never sufficient.

CRITICAL: n_trials MUST be the honest count of EVERY variant you tried — every
parameter, filter, pair, session, timeframe. Not 1. Garbage n_trials => garbage
Deflated Sharpe. Lying to this number only lies to you.

Run `python gauntlet.py` for the self-test (the math is checked on known-answer cases).
See example.py for an end-to-end demo on a real edge and a data-snooped fake one.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Optional, Sequence

EULER = 0.5772156649015329     # Euler-Mascheroni
E = math.e
DSR_CUTOFF = 0.95
MIN_TRADES = 400
MIN_REGIMES = 2
MIN_EPISODES = 3               # a declared specialist's regime must RECUR across >=3
                              # separated episodes (not all trades from one lucky stretch)
COST_RETENTION_FLOOR = 0.70


# --------------------------------------------------------------------------- #
# normal CDF / inverse CDF (no scipy)
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's algorithm, |err| < 1.15e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------- #
# moments
# --------------------------------------------------------------------------- #
def _moments(returns: Sequence[float]):
    """Return (T, mean, sample_std(ddof=1), skew, kurt[non-excess])."""
    T = len(returns)
    mu = statistics.fmean(returns)
    m2 = sum((x - mu) ** 2 for x in returns) / T          # population 2nd moment
    m3 = sum((x - mu) ** 3 for x in returns) / T
    m4 = sum((x - mu) ** 4 for x in returns) / T
    sd_pop = math.sqrt(m2) if m2 > 0 else 0.0
    sd_samp = math.sqrt(sum((x - mu) ** 2 for x in returns) / (T - 1)) if T > 1 else 0.0
    skew = (m3 / sd_pop ** 3) if sd_pop > 0 else 0.0
    kurt = (m4 / m2 ** 2) if m2 > 0 else 3.0              # non-excess (normal == 3)
    return T, mu, sd_samp, skew, kurt


def sharpe(returns: Sequence[float]) -> float:
    """Per-trade Sharpe = mean / sample-std (NOT annualised — PSR is scale-aware)."""
    T, mu, sd, _, _ = _moments(returns)
    return mu / sd if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# G3a  Probabilistic + Deflated Sharpe (Bailey & López de Prado)
# --------------------------------------------------------------------------- #
def probabilistic_sharpe(returns: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given sample length + skew + kurtosis."""
    T, mu, sd, skew, kurt = _moments(returns)
    if T < 3 or sd == 0:
        return 0.0
    sr = mu / sd
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return 0.0 if sr <= sr_benchmark else 1.0
    return norm_cdf((sr - sr_benchmark) * math.sqrt(T - 1) / math.sqrt(denom))


def deflated_sharpe(returns: Sequence[float], n_trials: int,
                    trial_sharpes: Optional[Sequence[float]] = None) -> dict:
    """Deflated Sharpe Ratio. n_trials = honest count of EVERY variant tried.
    trial_sharpes (optional) = the per-trade Sharpe of each variant; if given, the
    cross-trial variance is measured directly (the correct Bailey-LdP input). If not,
    it is estimated from the observed strategy's Sharpe sampling variance."""
    T, mu, sd, skew, kurt = _moments(returns)
    if T < 3 or sd == 0:
        return {"dsr": 0.0, "sr": 0.0, "sr0": 0.0, "psr0": 0.0, "n_trials": n_trials,
                "pass": False, "note": "degenerate (T<3 or zero variance)"}
    sr = mu / sd
    n = max(int(n_trials), 1)
    if trial_sharpes and len(trial_sharpes) >= 2:
        v_sr = statistics.variance(trial_sharpes)          # measured across variants
        v_src = "measured across trial_sharpes"
    else:
        # estimate Var(SR_hat) for THIS strategy (assume variants share it)
        v_sr = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / (T - 1)
        v_sr = max(v_sr, 1e-12)
        v_src = "estimated from sampling variance (no trial list supplied)"
    if n == 1:
        sr0 = 0.0                                          # no selection → benchmark 0
    else:
        sr0 = math.sqrt(v_sr) * ((1 - EULER) * norm_ppf(1 - 1.0 / n)
                                 + EULER * norm_ppf(1 - 1.0 / (n * E)))
    dsr = probabilistic_sharpe(returns, sr_benchmark=sr0)
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "psr0": probabilistic_sharpe(returns, 0.0),
            "n_trials": n, "v_src": v_src, "pass": dsr >= DSR_CUTOFF}


# --------------------------------------------------------------------------- #
# G3b  Block-bootstrap max drawdown (keeps clustered-loser autocorrelation)
# --------------------------------------------------------------------------- #
def max_drawdown(path: Sequence[float]) -> float:
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for r in path:
        eq += r
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd


def block_bootstrap_maxdd(returns: Sequence[float], block_lengths=None,
                          iters: int = 2000, pctile: float = 95.0, seed: int = 0) -> dict:
    """Circular block bootstrap of the equity path. Sweeps block length; reads the
    pctile of max drawdown at each. A trade-SHUFFLE (block=1) launders out the serial
    dependence that makes clustered losers; longer blocks preserve it. Reading the
    shape across block length tells you whether the DD tail is real."""
    T = len(returns)
    if T < 20:
        return {"hist_dd": max_drawdown(returns), "curve": {}, "verdict": "n<20: skip"}
    if block_lengths is None:
        hi = max(5, int(0.15 * T))
        block_lengths = sorted(set([1, 5, 10, 20, max(5, T // 20),
                                    max(10, T // 10), hi]))
        block_lengths = [L for L in block_lengths if 1 <= L <= hi]
    rng = random.Random(seed)
    curve = {}
    for L in block_lengths:
        dds = []
        nblocks = T // L + 1
        for _ in range(iters):
            path = []
            for _b in range(nblocks):
                s = rng.randrange(T)
                path.extend(returns[(s + k) % T] for k in range(L))
            dds.append(max_drawdown(path[:T]))
        dds.sort()
        curve[L] = dds[min(len(dds) - 1, int(pctile / 100.0 * len(dds)))]
    lo_L = min(curve); hi_L = max(curve)
    ratio = (curve[hi_L] / curve[lo_L]) if curve[lo_L] > 0 else float("inf")
    if ratio < 1.3:
        verdict = "FLAT plateau — DD tail is real, trust it"
    elif ratio < 2.0:
        verdict = "mild slope — tail slightly understated by a shuffle, size with margin"
    else:
        verdict = "STEEP climb — strategy rode a benign trade order; tail is much worse"
    return {"hist_dd": max_drawdown(returns), "curve": curve,
            "p95_short": curve[lo_L], "p95_long": curve[hi_L],
            "ratio": ratio, "pctile": pctile, "verdict": verdict}


# --------------------------------------------------------------------------- #
# G4a  Cost stress   G4b  Regime split
# --------------------------------------------------------------------------- #
def profit_factor(returns: Sequence[float]) -> float:
    gw = sum(r for r in returns if r > 0)
    gl = -sum(r for r in returns if r < 0)
    return gw / gl if gl > 0 else float("inf")


def cost_stress(returns: Sequence[float], extra_cost_pips: float) -> dict:
    """Subtract extra_cost_pips from every trade (≈ double-spread + a tick slippage)
    and require >=70% of expectancy retained AND no inversion (mean stays > 0)."""
    base = statistics.fmean(returns)
    stressed = [r - extra_cost_pips for r in returns]
    s_mean = statistics.fmean(stressed)
    if base <= 0:
        retention = 0.0
    else:
        retention = s_mean / base
    inverts = base > 0 and s_mean <= 0
    return {"extra_cost_pips": extra_cost_pips, "base_mean": base, "stressed_mean": s_mean,
            "retention": retention, "inverts": inverts,
            "base_pf": profit_factor(returns), "stressed_pf": profit_factor(stressed),
            "pass": (base > 0) and (not inverts) and (retention >= COST_RETENTION_FLOOR)}


def regime_split_pf(returns: Sequence[float], regimes: Sequence[str]) -> dict:
    """PF within each regime label; require PF>1 in every regime with n>=10."""
    by: dict[str, list] = {}
    for r, g in zip(returns, regimes):
        by.setdefault(g or "UNKNOWN", []).append(r)
    cells = {g: {"n": len(v), "pf": profit_factor(v), "pips": sum(v)} for g, v in by.items()}
    judged = {g: c for g, c in cells.items() if c["n"] >= 10}
    ok = bool(judged) and all(c["pf"] > 1.0 for c in judged.values())
    return {"cells": cells, "n_regimes": len(by), "pass": ok}


# --------------------------------------------------------------------------- #
# specialist helpers — a DECLARED regime specialist (e.g. a trend strategy gated to
# TREND_UP/DOWN) must NOT be required to profit in regimes it never trades. We judge
# it ONLY within its declared regime(s), but across TIME: does the edge hold over
# several SEPARATED episodes of that regime, or did it all come from one lucky stretch?
# --------------------------------------------------------------------------- #
def regime_episodes(regimes: Sequence[str], times: Optional[Sequence[str]],
                    declared: set) -> int:
    """Count contiguous in-declared-regime episodes over time."""
    order = sorted(range(len(regimes)), key=lambda i: times[i]) if times else range(len(regimes))
    eps = 0
    prev_in = False
    for i in order:
        cur = regimes[i] in declared
        if cur and not prev_in:
            eps += 1
        prev_in = cur
    return eps


def time_split_pf(returns: Sequence[float], times: Optional[Sequence[str]], k: int = 3) -> dict:
    """Split the (time-ordered) in-regime trades into k contiguous blocks; require
    PF>1 in each block with n>=10. The specialist replacement for the cross-regime
    split: stability across the regime's recurrences over TIME, not across regimes."""
    order = sorted(range(len(returns)), key=lambda i: times[i]) if times else list(range(len(returns)))
    ordered = [returns[i] for i in order]
    m = len(ordered)
    blocks = [ordered[j * m // k:(j + 1) * m // k] for j in range(k)]
    cells = [{"n": len(b), "pf": profit_factor(b), "pips": sum(b)} for b in blocks]
    judged = [c for c in cells if c["n"] >= 10]
    ok = bool(judged) and all(c["pf"] > 1.0 for c in judged)
    return {"cells": cells, "pass": ok, "k": k}


# --------------------------------------------------------------------------- #
# G5  Parameter-sensitivity plateau (robust edge vs sharp overfit peak)
# --------------------------------------------------------------------------- #
def parameter_plateau(scores: dict, retain: float = 0.6, min_keep_frac: float = 0.6) -> dict:
    """Is the edge a robust PLATEAU or a fragile overfit PEAK?

    scores: {param_value: metric} from a 1-D sweep of ONE knob (lookback, stop, threshold…),
    metric = higher-is-better (Sharpe, PF-1, expectancy). A REAL edge degrades GRACEFULLY as
    you step off the chosen value — its grid neighbors keep most of the peak. An OVERFIT edge
    is a spike: great at the exact value tried, dead one step away (classic curve-fit tell).

    Rule: PLATEAU iff (a) the best setting's immediate neighbors each retain >= `retain` of the
    peak AND (b) >= `min_keep_frac` of the whole grid does too. Anything less = ridge / peak.
    Caller runs the backtest at each param and passes the resulting metrics (keeps this module
    results-only, no backtest dependency). Use the SAME data slice for every point."""
    if len(scores) < 3:
        return {"verdict": "GRID TOO SMALL (need >=3 points)", "pass": False, "scores": dict(scores)}
    items = sorted(scores.items(), key=lambda kv: kv[0])
    params = [k for k, _ in items]
    vals = [v for _, v in items]
    peak = max(vals)
    ip = vals.index(peak)
    peak_param = params[ip]
    if peak <= 0:
        return {"peak": peak, "peak_param": peak_param, "keep_frac": 0.0, "neighbors": [],
                "neighbors_ok": False, "verdict": "NO EDGE (peak <= 0)", "pass": False,
                "scores": dict(items)}
    thresh = retain * peak
    keep_frac = sum(1 for v in vals if v >= thresh) / len(vals)
    neigh = ([vals[ip - 1]] if ip - 1 >= 0 else []) + ([vals[ip + 1]] if ip + 1 < len(vals) else [])
    neighbors_ok = bool(neigh) and all(v >= thresh for v in neigh)
    if keep_frac >= min_keep_frac and neighbors_ok:
        verdict = "PLATEAU (robust)"
    elif keep_frac >= 0.4 or neighbors_ok:
        verdict = "RIDGE (borderline — size with margin)"
    else:
        verdict = "PEAK (overfit risk — edge dies off-parameter)"
    return {"peak": peak, "peak_param": peak_param, "thresh": thresh, "retain": retain,
            "keep_frac": keep_frac, "neighbors": neigh, "neighbors_ok": neighbors_ok,
            "verdict": verdict, "pass": verdict.startswith("PLATEAU"), "scores": dict(items)}


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_gates(returns: Sequence[float], n_trials: int, *, mechanism: str = "",
              regimes: Optional[Sequence[str]] = None, times: Optional[Sequence[str]] = None,
              declared_regimes: Optional[set] = None, extra_cost_pips: float = 1.0,
              trial_sharpes: Optional[Sequence[float]] = None,
              wf_positive: Optional[tuple] = None, label: str = "candidate",
              bb_iters: int = 2000, quiet: bool = False) -> dict:
    """Cost-ordered gauntlet. Returns a verdict dict.

    declared_regimes: if the strategy is a DECLARED specialist (live allowed_regimes),
      pass that set. We then judge it ONLY within those regimes (all gates run on the
      in-regime subset), the sample floor counts IN-REGIME trades + requires the regime
      to RECUR across >=MIN_EPISODES separated episodes, and the cross-regime split is
      replaced by a TIME split within the regime. A specialist is NOT penalised for not
      working in markets it never trades — but it must clear a higher in-regime bar.
    wf_positive: (n_pass, n_windows) if walk-forward was run elsewhere."""
    n_all = len(returns)
    specialist = bool(declared_regimes) and bool(regimes)
    if specialist:
        order = sorted(range(n_all), key=lambda i: (times[i] if times else 0))
        keep = [i for i in order if regimes[i] in declared_regimes]
        eval_ret = [returns[i] for i in keep]
        eval_times = [times[i] for i in keep] if times else None
        episodes = regime_episodes(regimes, times, declared_regimes) if times else None
    else:
        eval_ret = list(returns)
        eval_times = times
        episodes = None
    n = len(eval_ret)

    print(f"\n{'='*70}\nKILL-GATES: {label}   (n={n} trades, n_trials={n_trials})\n{'='*70}")
    if specialist:
        ep_str = f", episodes={episodes}" if episodes is not None else ""
        print(f"  SPECIALIST mode — judged ONLY within {sorted(declared_regimes)} "
              f"({n}/{n_all} trades in-regime{ep_str})")
    print(f"G1 MECHANISM (manual — name the loser & why they keep losing):")
    print(f"    {mechanism or '*** NOT STATED — do not risk capital until you can ***'}")

    # G2 sample floor (in-regime n for specialists; episode-diversity replaces regime-diversity)
    if specialist:
        ep_ok = (episodes is None) or (episodes >= MIN_EPISODES)
        g2 = {"n": n, "episodes": episodes, "pass": n >= MIN_TRADES and ep_ok}
        print(f"\nG2 SAMPLE FLOOR (in-regime): n={n} (need >={MIN_TRADES}), "
              f"episodes={episodes} (need >={MIN_EPISODES}) -> {'PASS' if g2['pass'] else 'FAIL'}")
    else:
        n_reg = len(set(regimes)) if regimes else 0
        g2 = {"n": n, "n_regimes": n_reg,
              "pass": n >= MIN_TRADES and (n_reg >= MIN_REGIMES if regimes else True)}
        reg_str = f", regimes={n_reg} (need >={MIN_REGIMES})" if regimes else " (no regime tags)"
        print(f"\nG2 SAMPLE FLOOR: n={n} (need >={MIN_TRADES}){reg_str} "
              f"-> {'PASS' if g2['pass'] else 'FAIL'}")
    if n < MIN_TRADES:
        print(f"    too small to tell skill from luck; metrics below are INDICATIVE only.")

    # G3a deflated sharpe (on the evaluation set)
    ds = deflated_sharpe(eval_ret, n_trials, trial_sharpes)
    print(f"\nG3a DEFLATED SHARPE: SR={ds['sr']:+.3f}  SR0(maxof{ds['n_trials']})="
          f"{ds['sr0']:.3f}  PSR(vs0)={ds.get('psr0',0):.3f}  "
          f"DSR={ds['dsr']:.3f} (need >={DSR_CUTOFF}) -> {'PASS' if ds['pass'] else 'FAIL'}")
    print(f"    [{ds.get('v_src','')}]")

    # G3b block bootstrap DD
    bb = block_bootstrap_maxdd(eval_ret, iters=bb_iters)
    if bb["curve"]:
        curve_str = "  ".join(f"L{L}:{dd:.0f}p" for L, dd in bb["curve"].items())
        print(f"\nG3b BLOCK-BOOTSTRAP maxDD p{bb['pctile']:.0f}: hist={bb['hist_dd']:.0f}p")
        print(f"    {curve_str}")
        print(f"    ratio(long/short)={bb['ratio']:.2f} -> {bb['verdict']}")

    # G4a cost stress
    cs = cost_stress(eval_ret, extra_cost_pips) if eval_ret else {"pass": False}
    if eval_ret:
        print(f"\nG4a COST STRESS (+{extra_cost_pips}p/trade): mean {cs['base_mean']:+.2f}p"
              f" -> {cs['stressed_mean']:+.2f}p  retention={cs['retention']*100:.0f}%"
              f" (need >={COST_RETENTION_FLOOR*100:.0f}%)  inverts={cs['inverts']}"
              f" -> {'PASS' if cs['pass'] else 'FAIL'}")

    # G4b  generalist: cross-regime split | specialist: TIME split within the regime
    rs = None
    if specialist:
        rs = time_split_pf(eval_ret, eval_times, k=3)
        cells = "  ".join(f"T{j+1}:PF{c['pf']:.2f}(n{c['n']})" for j, c in enumerate(rs["cells"]))
        print(f"\nG4b TIME SPLIT within regime (PF>1 in each n>=10 third): -> "
              f"{'PASS' if rs['pass'] else 'FAIL'}\n    {cells}")
    elif regimes:
        rs = regime_split_pf(returns, regimes)
        cells = "  ".join(f"{g}:PF{c['pf']:.2f}(n{c['n']})" for g, c in
                          sorted(rs["cells"].items(), key=lambda kv: -kv[1]["n"]))
        print(f"\nG4b REGIME SPLIT (PF>1 in every n>=10 cell): -> "
              f"{'PASS' if rs['pass'] else 'FAIL'}\n    {cells}")

    # walk-forward (if supplied)
    wf_pass = None
    if wf_positive:
        wp, wn = wf_positive
        wf_pass = wp >= max(1, round(0.7 * wn))
        print(f"\nG3c WALK-FORWARD (supplied): {wp}/{wn} windows positive (need >=70%) "
              f"-> {'PASS' if wf_pass else 'FAIL'}")

    gates = {"G2_sample": g2["pass"], "G3a_dsr": ds["pass"], "G4a_cost": cs["pass"]}
    if rs is not None:
        gates["G4b_split"] = rs["pass"]
    if wf_pass is not None:
        gates["G3c_wf"] = wf_pass
    survived = all(gates.values())
    print(f"\n{'-'*70}\nVERDICT [{label}]: {'SURVIVES ✓' if survived else 'KILLED ✗'}  "
          f"({sum(gates.values())}/{len(gates)} automated gates) "
          f"{'' if survived else '— ' + ', '.join(k for k,v in gates.items() if not v)}")
    print("(G1 mechanism is yours to certify — automated PASS is necessary, not sufficient.)")
    return {"label": label, "survived": survived, "gates": gates, "specialist": specialist,
            "n_in_regime": n, "episodes": episodes,
            "deflated_sharpe": ds, "block_bootstrap": bb, "cost_stress": cs,
            "regime_split": rs, "g2": g2, "wf_pass": wf_pass}


# --------------------------------------------------------------------------- #
# self-test: the math must be right or this whole tool is worse than nothing
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    rng = random.Random(7)
    print("### SELFTEST ###")

    # 1) PSR of EXACTLY-zero-mean noise == 0.5; and E[PSR] over many raw-noise
    #    samples ~ 0.5 (a single sample has a nonzero sample mean, so its PSR is not 0.5)
    noise = [rng.gauss(0, 1) for _ in range(1000)]
    dm = statistics.fmean(noise)
    p_demeaned = probabilistic_sharpe([x - dm for x in noise], 0.0)
    avg = statistics.fmean(
        probabilistic_sharpe([rng.gauss(0, 1) for _ in range(500)], 0.0) for _ in range(40))
    print(f"1) PSR(demeaned noise)={p_demeaned:.3f} (expect 0.50)  "
          f"E[PSR over 40 noise samples]={avg:.3f} (expect ~0.50)  "
          f"{'OK' if abs(p_demeaned-0.5)<0.02 and 0.42<avg<0.58 else 'FAIL'}")

    # 2) strong real edge, ONE trial -> DSR high
    strong = [rng.gauss(0.20, 1.0) for _ in range(600)]
    d2 = deflated_sharpe(strong, n_trials=1)
    print(f"2) DSR(strong edge, 1 trial) = {d2['dsr']:.3f}  (expect high)  "
          f"{'OK' if d2['dsr'] > 0.9 else 'FAIL'}")

    # 3) SAME strong edge but admit it was the best of 200 tries -> DSR deflates
    d3 = deflated_sharpe(strong, n_trials=200)
    print(f"3) DSR(same edge, 200 trials) = {d3['dsr']:.3f}  SR0={d3['sr0']:.3f}  "
          f"(expect < #2)  {'OK' if d3['dsr'] < d2['dsr'] else 'FAIL'}")

    # 4) weak edge best-of-many -> should FAIL the 0.95 cutoff
    weak = [rng.gauss(0.04, 1.0) for _ in range(450)]
    d4 = deflated_sharpe(weak, n_trials=100)
    print(f"4) DSR(weak edge, 100 trials) = {d4['dsr']:.3f}  "
          f"(expect <0.95 FAIL)  {'OK' if not d4['pass'] else 'FAIL'}")

    # 5) block bootstrap should see AUTOCORRELATION: an AR(1) series (clustered moves)
    #    must show a steeper long/short ratio than an i.i.d. series with no clustering.
    iid = [rng.gauss(0.0, 1.0) for _ in range(600)]
    ar = []
    prev = 0.0
    for _ in range(600):
        prev = 0.6 * prev + rng.gauss(-0.05, 1.0)        # positive serial correlation
        ar.append(prev)
    bb_iid = block_bootstrap_maxdd(iid, iters=800)
    bb_ar = block_bootstrap_maxdd(ar, iters=800)
    print(f"5) block-bootstrap  iid ratio={bb_iid['ratio']:.2f}  AR(1) ratio={bb_ar['ratio']:.2f}"
          f"  (expect AR>iid)  {'OK' if bb_ar['ratio'] > bb_iid['ratio'] else 'FAIL'}")
    print(f"     iid  -> {bb_iid['verdict']}")
    print(f"     AR(1)-> {bb_ar['verdict']}")

    # 6) cost stress inverts a razor-thin edge
    thin = [rng.gauss(0.3, 2.0) for _ in range(300)]
    cs = cost_stress(thin, extra_cost_pips=0.5)
    print(f"6) cost-stress thin edge: {cs['base_mean']:+.2f} -> {cs['stressed_mean']:+.2f}p "
          f"retention={cs['retention']*100:.0f}% inverts={cs['inverts']}")

    # 7) parameter plateau: a smooth hill PASSES; a lone spike FAILS
    plateau = {8: 0.50, 10: 0.55, 12: 0.58, 14: 0.56, 16: 0.49}
    peak = {8: 0.05, 10: 0.09, 12: 0.58, 14: 0.07, 16: 0.03}
    pp_ok = parameter_plateau(plateau)
    pp_bad = parameter_plateau(peak)
    print(f"7) param-plateau: hill -> {pp_ok['verdict']} (keep {pp_ok['keep_frac']*100:.0f}%)  |  "
          f"spike -> {pp_bad['verdict']} (keep {pp_bad['keep_frac']*100:.0f}%)  "
          f"{'OK' if pp_ok['pass'] and not pp_bad['pass'] else 'FAIL'}")
    print("### SELFTEST DONE ###")


if __name__ == "__main__":
    _selftest()
