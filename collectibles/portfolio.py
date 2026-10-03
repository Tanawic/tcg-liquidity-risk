"""Portfolio-level TCG liquidity risk: build PositionRisk per holding, aggregate,
and run stress scenarios.  Inputs are local files only (``collectibles.data``); no network access.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from collectibles.data import load_listings, load_weekly
from collectibles.liquidity import (
    Demand, PositionRisk, amihud, bangia_lvar, clean_ladder, cornish_fisher_var,
    days_to_liquidate, fit_demand, ladder_from_listings, liquidate, scale_ladder,
)

log = logging.getLogger("collectibles.portfolio")

HORIZONS = (1, 7, 30, 90)
ALPHAS = (0.50, 0.05)          # typical week vs stressed (5th-pct demand)
MIN_TRADE_WEEKS = 13           # fewer weeks with a sale → price VaR flagged unreliable
MAX_HAIRCUT = 0.20             # "liquid" = can exit at ≤ 20% haircut (fees alone ≈ 14%)
DUMP_FRAC = 0.60               # outside option: shop / buylist bid as a fraction of market

# Hypothetical scenarios (not calibrated to a specific episode):
#   demand_crunch — hype unwinds: buyers vanish, holders rush to list
#   reprint       — supply shock on reprint-exposed items only
SCENARIOS = {
    "demand_crunch": {"lam": 0.40, "price": 0.75, "depth": 1.5, "only_flagged": False},
    "reprint":       {"lam": 0.70, "price": 0.65, "depth": 3.0, "only_flagged": True},
}


def analyse_position(pos: dict, data_dir: Path) -> tuple[PositionRisk, pd.DataFrame, list]:
    iid = str(pos["id"])
    q = int(pos.get("qty", 1))
    game = pos.get("game", "")

    wk = load_weekly(data_dir, iid)
    raw = ladder_from_listings(load_listings(data_dir, iid))
    market = float(wk["market"].iloc[-1]) if len(wk) and not math.isnan(wk["market"].iloc[-1])         else float(pos.get("market_override", float("nan")))
    ladder = clean_ladder(raw, market) if market > 0 else raw
    demand = fit_demand(wk["qty"].tolist()) if len(wk) else Demand(0.0, 1.0)
    if "dump_frac" in pos:
        dump, bid_src = float(pos["dump_frac"]), "position"
    else:
        dump, bid_src = DUMP_FRAC, "assumed"

    depth = sum(l.qty for l in ladder)
    pr = PositionRisk(
        name=pos.get("name") or iid, qty=q, market=market, value=q * market,
        lam_day=demand.lam_day, phi=demand.phi, book_depth=depth,
        days_of_supply=depth / demand.lam_day if demand.lam_day > 0 else float("inf"),
        best_ask=ladder[0].landed if ladder else float("nan"),
        dump_frac=dump, bid_source=bid_src, game=game,
        trade_weeks=int((wk["txn"] > 0).sum()) if len(wk) else 0,
    )
    if not (market > 0):
        log.warning("%s: no market price, skipped", pr.name)
        return pr, wk, ladder

    for h in HORIZONS:
        for a in ALPHAS:
            pr.liq[(h, a)] = liquidate(ladder, market, q, demand, h, a, dump)
    pr.days_to_liq = days_to_liquidate(ladder, market, q, demand, MAX_HAIRCUT, dump_frac=dump)
    pr.bangia = bangia_lvar(wk["ret"].to_numpy(), wk["spread"].to_numpy(), 0.95, 4)
    pr.amihud = amihud(wk["ret"].to_numpy(), wk["dvol"].to_numpy())

    hc30 = pr.liq[(30, 0.05)].haircut
    if not math.isnan(pr.bangia.price_var):
        pr.combined_30d = 1 - (1 - pr.bangia.price_var) * (1 - hc30)
    return pr, wk, ladder


@dataclass
class PortfolioRisk:
    positions: list[PositionRisk]
    mtm: float
    liq_value: dict            # (h, α) → $ proceeds
    price_var_4w: float        # portfolio price VaR 95% (diversified), 4 weeks
    combined_30d: float
    stress: dict               # scenario → {"value": $, "loss": %}
    weekly: dict               # name → weekly frame
    ladders: dict              # name → ladder


def run(portfolio: list[dict], data_dir: Path) -> PortfolioRisk:
    pos, weekly, ladders, inputs = [], {}, {}, []
    for p in portfolio:
        try:
            pr, wk, lad = analyse_position(p, Path(data_dir))
        except Exception as e:  # one bad product must not kill the report
            log.warning("position %s failed: %s", p.get("id"), e)
            continue
        if pr.market > 0:
            pos.append(pr)
            weekly[pr.name], ladders[pr.name] = wk, lad
            inputs.append(p)

    mtm = sum(p.value for p in pos)
    liq_value = {k: sum(p.liq[k].proceeds for p in pos) for k in pos[0].liq} if pos else {}

    # diversified price VaR from aligned weekly simple returns
    rets = pd.concat({p.name: weekly[p.name].set_index("date")["ret"] for p in pos}, axis=1).dropna()
    w = np.array([p.value / mtm for p in pos]) if mtm else np.array([])
    pvar = float("nan")
    if len(rets) >= 8 and w.size:
        port = np.log1p((np.expm1(rets.to_numpy()) * w).sum(axis=1))
        pvar = 1 - math.exp(-cornish_fisher_var(port, 0.95, horizon=4))
    hc30 = 1 - liq_value.get((30, 0.05), 0) / mtm if mtm else float("nan")
    combined = 1 - (1 - pvar) * (1 - hc30) if not math.isnan(pvar) else float("nan")

    stress = {}
    for name, s in SCENARIOS.items():
        total = 0.0
        for p, inp in zip(pos, inputs):
            hit = inp.get("reprint_risk", False) or not s["only_flagged"]
            if not hit:
                total += p.liq[(30, 0.05)].proceeds
                continue
            d = Demand(p.lam_day * s["lam"], p.phi)
            lad = scale_ladder(ladders[p.name], s["price"], s["depth"])
            total += liquidate(lad, p.market * s["price"], p.qty, d, 30, 0.05, p.dump_frac).proceeds
        stress[name] = {"value": total, "loss": 1 - total / mtm if mtm else float("nan")}

    return PortfolioRisk(pos, mtm, liq_value, pvar, combined, stress, weekly, ladders)
