# Gauntlet

**An honest backtest validation framework — its job is to *reject* your strategy, not to bless it.**

A real trading edge survives friction and scrutiny. An overfit one collapses the moment either touches it. Gauntlet runs your strategy's per-trade returns through a sequence of statistical kill-tests, cheapest-rejection-first, and tells you where it dies.

One file, pure Python standard library — no numpy, no pandas, nothing to install. You bring a list of per-trade returns; it brings the skepticism.

---

## Why this exists

I spent about a year building forex trading bots. Strategy after strategy looked great in backtest and then lost money — or went nowhere — live. The problem was never a shortage of strategies. It was that I kept *fooling myself*: mining dozens of variants, keeping the best-looking one, and mistaking a lucky stretch for an edge.

Gauntlet is the tool I built to stop doing that. It is deliberately pessimistic. Most of my own strategies do not survive it, and that is the point — a backtest that can't fail you can't protect you either. The full story is on the [journey page](https://thrasherapps.com/journey.html).

---

## Quickstart

No dependencies. Python 3.8+.

```bash
git clone https://github.com/Cbthrasher/Gauntlet.git
cd Gauntlet
python example.py     # a 60-second guided tour (real edge vs. data-snooped fake)
python gauntlet.py    # run the self-test — the math is verified on known answers
```

Run it on your own strategy:

```python
import gauntlet as g

# one number per trade: pips, dollars, R — whatever you measure in
returns = [...]

g.run_gates(
    returns,
    n_trials=37,                  # HONEST count of every variant you tried (see below)
    mechanism="late breakout-chasers who are systematically on the wrong side",
    regimes=my_regime_labels,     # optional: one label per trade, e.g. "CALM"/"WILD"
    extra_cost_pips=1.0,          # the friction to stress-test against
)
```

It prints each gate's verdict and a final **SURVIVES** / **KILLED**.

---

## The gates

Run cheapest-rejection-first, so a hopeless strategy dies fast and cheap:

| Gate | Name | What it asks |
|------|------|--------------|
| **G1** | Mechanism | *Who is on the other side of this trade, and why do they keep losing?* You answer this — it can't be automated, and an automated pass is necessary but never sufficient. |
| **G2** | Sample floor | Are there enough trades (≥400) across ≥2 volatility regimes to tell skill from luck? |
| **G3a** | **Deflated Sharpe** | Adjusts the Sharpe for **how many variants you tried** (plus skew and kurtosis) and returns the probability the edge is real. Cutoff 0.95. *This is the gate almost nobody runs, and the one that catches data-snooping.* |
| **G3b** | Block-bootstrap max drawdown | Resamples return *blocks* (not single trades) so clustered losers keep their autocorrelation — revealing the true drawdown tail that a naive shuffle launders away. |
| **G4a** | Cost stress | Add slippage/cost to every trade. A real edge keeps ≥70% of its expectancy; a fake one flips sign the instant friction appears. |
| **G4b** | Regime split | Profitable in *both* the calm and the stressed half (or, for a declared specialist, across separated episodes of its own regime). |
| **G5** | Parameter plateau | Sweep one knob: a real edge is a robust *plateau* (its neighbors still work); an overfit one is a sharp *peak* that dies one step away. |

---

## The one rule that makes it work

> `n_trials` must be the honest count of **every** variant you tried — every parameter, filter, pair, session, and timeframe.

The Deflated Sharpe (Bailey & López de Prado, 2014) exists because if you try 300 strategies and keep the best, the winner looks great *by chance alone*. Telling the gauntlet `n_trials=1` when you really tried 300 doesn't fool the gauntlet — it only fools you.

`example.py` shows the exact same trades **passing** at `n_trials=1` and getting **killed** at `n_trials=300`. Nothing about the strategy changed. Only your honesty did.

---

## What it is, and what it isn't

- **It is** a referee for returns you already have. Hand it a flat list of per-trade results and it judges them.
- **It is not** a backtester. It does not fetch data, generate signals, or simulate fills — that is your job, and doing it without look-ahead bias is the other half of the battle.
- **It will not** tell you a strategy is good. The strongest thing it ever says is *"I couldn't kill this one."* That is also the strongest honest claim any backtest can make.

---

## Credits

- Probabilistic & Deflated Sharpe Ratio: David H. Bailey & Marcos López de Prado.
- The gate framework grew out of a lot of losing trades and a lot of community wisdom. The long-form story is on the [journey page](https://thrasherapps.com/journey.html).

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, sell it. If it talks you out of one overfit strategy, it has already paid for itself.
