"""TCG liquidity-risk report for a portfolio of trading cards, from local data files.

Usage:
    python scripts/tcg_liquidity.py                                     # synthetic sample in data/sample/
    python scripts/tcg_liquidity.py --portfolio my_data/portfolio.json  # your own dataset (same layout)

The dataset folder is the one that holds portfolio.json (layout: collectibles/data.py).
No network access: the model only reads files.

Outputs: outputs/tcg_liquidity/latest_<dataset>.{md,json}, haircut_curves_<dataset>.png
Model  : collectibles/liquidity.py (queue liquidation + Bangia LVaR)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np                                                  # noqa: E402
from collectibles.liquidity import Demand, liquidate                  # noqa: E402
from collectibles.portfolio import ALPHAS, DUMP_FRAC, HORIZONS, MAX_HAIRCUT, MIN_TRADE_WEEKS, SCENARIOS, run  # noqa: E402

OUT = ROOT / "outputs" / "tcg_liquidity"


def pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def usd(x: float) -> str:
    return f"${x:,.0f}"


def chart(r, path: Path) -> None:
    import matplotlib.pyplot as plt
    from research.quant_style import C, apply, watermark
    apply()
    colors = [C.BLUE, C.RED, C.GREEN, C.AMBER, C.PURPLE, C.TEAL, C.DARK, C.GRAY]
    hs = list(range(1, 121))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, p in enumerate(r.positions):
        lad = r.ladders[p.name]
        d = Demand(p.lam_day, p.phi)
        hc = [liquidate(lad, p.market, p.qty, d, h, 0.05, p.dump_frac).haircut for h in hs]
        a1.plot(hs, hc, color=colors[i % len(colors)], lw=1.6, label=f"{p.name} ×{p.qty}")
    a1.axhline(MAX_HAIRCUT, color=C.GRAY, ls="--", lw=1)
    bid = float(np.median([p.dump_frac for p in r.positions]))
    src = "assumed" if all(p.bid_source == "assumed" for p in r.positions) else "median, calibrated"
    a1.axhline(1 - bid, color=C.RED, ls=":", lw=1)
    a1.text(3, 1 - bid + 0.01, f"shop bid {bid:.0%} of market ({src}, sell today)", ha="left", fontsize=8, color=C.RED)
    a1.text(118, MAX_HAIRCUT + 0.01, f"liquid threshold {MAX_HAIRCUT:.0%}", ha="right", fontsize=8, color=C.GRAY)
    a1.set(xlabel="liquidation horizon (days)", ylabel="haircut vs market (stressed, 5th-pct demand)",
           title="Price–time frontier per holding")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_ylim(0, 0.5)
    a1.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.88))

    labels = ["MTM", "30d median", "30d stressed"] + list(r.stress)
    vals = [r.mtm, r.liq_value[(30, 0.5)], r.liq_value[(30, 0.05)]] + [s["value"] for s in r.stress.values()]
    bars = a2.bar(labels, vals, color=[C.DARK, C.BLUE, C.AMBER, C.RED, C.PURPLE])
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v, f"{1 - v / r.mtm:.0%}" if v != r.mtm else usd(v),
                ha="center", va="bottom", fontsize=8)
    a2.set(title="Portfolio liquidation value", ylabel="USD")
    a2.tick_params(axis="x", labelsize=8, rotation=15)
    watermark(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def flag(p) -> str:
    return "*" if p.trade_weeks < MIN_TRADE_WEEKS else ""


def report(r, today: str, chart_name: str = "haircut_curves.png") -> str:
    L = [f"# TCG Liquidity Risk — {today}", "",
         f"{len(r.positions)} holdings. Mark-to-market (latest market price × qty) **{usd(r.mtm)}**", "",
         "## Portfolio", "",
         "| Metric | Value |", "|---|---|",
         f"| Liquidation value, 30d median demand | {usd(r.liq_value[(30, 0.5)])} ({pct(1 - r.liq_value[(30, 0.5)] / r.mtm)} haircut) |",
         f"| Liquidation value, 30d stressed (5th-pct demand) | {usd(r.liq_value[(30, 0.05)])} ({pct(1 - r.liq_value[(30, 0.05)] / r.mtm)} haircut) |",
         f"| Sell everything today (shop bid) | {usd(r.liq_value[(1, 0.05)])} |",
         f"| Price VaR 95%, 4w (diversified) | {pct(r.price_var_4w)} |",
         f"| **Combined 30d loss** = 1−(1−VaR)(1−haircut) | **{pct(r.combined_30d)}** |"]
    for k, s in r.stress.items():
        c = SCENARIOS[k]
        desc = (f"demand ×{c['lam']}, price ×{c['price']}, book depth ×{c['depth']}"
                + (", flagged only" if c["only_flagged"] else ""))
        L.append(f"| Stress `{k}` ({desc}), 30d | {usd(s['value'])} ({pct(s['loss'])} loss) |")
    L += ["", "## Positions", "",
          "| Holding | Qty | Market | Best ask | Sold/day | Book depth | Days of supply | "
          "Haircut 7d | Haircut 30d (med / stress) | Days to ≤20% | Price VaR 4w | Bangia LVaR | Amihud | Shop bid |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(r.positions, key=lambda p: -p.value):
        L.append(
            f"| {p.name} | {p.qty} | {usd(p.market)} | {usd(p.best_ask) if p.best_ask == p.best_ask else '–'} | "
            f"{p.lam_day:.2f} | {p.book_depth} | {p.days_of_supply:.0f} | "
            f"{pct(p.liq[(7, 0.5)].haircut)} | {pct(p.liq[(30, 0.5)].haircut)} / {pct(p.liq[(30, 0.05)].haircut)} | "
            f"{p.days_to_liq if p.days_to_liq else '>365'} | {pct(p.bangia.price_var)}{flag(p)} | {pct(p.bangia.lvar)}{flag(p)} | "
            f"{p.amihud:.4f} | {p.dump_frac:.0%} ({p.bid_source}) |")
    L += ["", f"![haircut curves]({chart_name})", "",
          "## How to read", "",
          "- **Haircut** = 1 − net cash / (qty × market price), after marketplace fees (13.25% + $0.30) and shipping. "
          "Even selling exactly at market costs ~14%: that is the fee floor.",
          "- **Queue model**: there is no bid. To sell q units within h days you must price so that at most D(h) − q "
          "units are listed ahead of you. D(h) = demand in h days at the 50th (median) or 5th (stressed) percentile "
          "of a Negative Binomial fitted to weekly sales.",
          f"- **Shop bid** = the outside option, {DUMP_FRAC:.0%} of market by default (override per holding with "
          "`dump_frac`). If undercutting would net less than this, the model sells to the shop instead.",
          "- **Days of supply** = units listed ÷ units sold per day. The higher it is, the longer the queue.",
          "- **Bangia LVaR** = price VaR + ½(μ_S + a·σ_S) of the weekly sale-price range; a is read off the 95th "
          "percentile of observed spreads (at least 1.645). Upward drift is not counted. Independent of position size.",
          "- Asks below 60% of market are dropped as stale or mislabelled listings.",
          f"- `*` = fewer than {MIN_TRADE_WEEKS} of 52 weeks had a sale, so the market price barely moves and its VaR "
          "is unreliable; use the queue-model haircut instead.", "",
          "Limitations: the queue is held fixed over the horizon, only one marketplace's demand is counted, and the "
          "stress scenarios are hypothetical."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--portfolio", default=str(ROOT / "data" / "sample" / "portfolio.json"),
                    help="portfolio.json inside a dataset folder (weekly/ and listings/ next to it)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    holdings = json.loads(Path(a.portfolio).read_text(encoding="utf-8"))["positions"]
    r = run(holdings, Path(a.portfolio).parent)
    if not r.positions:
        print("no position could be priced — check the dataset files")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    sfx = f"_{Path(a.portfolio).resolve().parent.name}"
    chart_name = f"haircut_curves{sfx}.png"
    chart(r, OUT / chart_name)
    md = report(r, today, chart_name)
    (OUT / f"latest{sfx}.md").write_text(md, encoding="utf-8")
    (OUT / f"latest{sfx}.json").write_text(json.dumps({
        "date": today, "mtm": r.mtm, "price_var_4w": r.price_var_4w, "combined_30d": r.combined_30d,
        "liq_value": {f"{h}d_a{a_}": r.liq_value[(h, a_)] for h in HORIZONS for a_ in ALPHAS},
        "stress": r.stress,
        "positions": [{
            "name": p.name, "game": p.game, "qty": p.qty, "market": p.market, "best_ask": p.best_ask,
            "lam_day": p.lam_day, "phi": p.phi, "book_depth": p.book_depth,
            "days_of_supply": p.days_of_supply, "days_to_liq": p.days_to_liq,
            "haircut": {f"{h}d_a{a_}": p.liq[(h, a_)].haircut for h in HORIZONS for a_ in ALPHAS},
            "trade_weeks": p.trade_weeks, "price_var_4w": p.bangia.price_var, "bangia_a": p.bangia.a, "bangia_lvar": p.bangia.lvar, "amihud": p.amihud,
            "dump_frac": p.dump_frac, "bid_source": p.bid_source,
        } for p in r.positions],
    }, indent=2, default=float), encoding="utf-8")
    print(md)
    print(f"→ {OUT / f'latest{sfx}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
