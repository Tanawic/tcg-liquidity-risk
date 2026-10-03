"""Liquidity-risk model for trading-card holdings (TCGplayer: Pokémon, One Piece).

Thesis (why liquidity is the risk that matters for a TCG holder)
----------------------------------------------------------------
TCGplayer is an ask-only market: there is no resting bid, so a seller joins a
price-sorted queue of listings and waits for buyers, who fill from the cheapest
landed price (price + shipping) upward.  "Liquidating" therefore means choosing
a point on a price–time frontier: undercut deep into the queue and sell fast, or
list near market and wait.  The counterparty who absorbs a forced seller is the
marginal buyer at the bottom of the ladder (or a shop buylist at a steep
discount).  Liquidity dries up exactly when you need it — hype fades, a reprint
lands, a new set pulls money away — because demand falls AND the ladder thickens
as other holders rush to list.

Model layers
------------
1. Demand      weekly units sold → daily rate λ (recent window) + dispersion φ
               (var/mean of weekly counts) → units demanded in h days
               D_h ~ NegBin(mean λh, var φλh).
2. Queue       to sell q units within h days with confidence (1−α) you must sit
               ahead of at most D_h,α − q units of the ask ladder → price p*.
               Unfilled units — or all units, if undercutting nets less — go to
               the outside option: a shop/buylist bid at ``dump_frac``·market.
               Proceeds net of marketplace fees and own shipping → haircut(q, h).
3. Exogenous   Bangia et al. (1999) LVaR: price VaR (Cornish–Fisher on weekly
               log returns; drift ×H, vol ×√H) + ½(μ_S + a·σ_S) with S = weekly
               (high−low)/market sale-price range as a spread proxy and a read
               off the empirical spread tail (≥ normal z).
4. Diagnostics Amihud ILLIQ = mean |r| / $volume;  days of supply = book / λ.

Combined loss at horizon h:  1 − (1 − VaR_h)·(1 − haircut_h,α).

Known simplifications (all conservative except the first):
* the ladder is static over h (no new competing listings arrive) — the stress
  scenarios thicken it to compensate;
* only TCGplayer demand counts (eBay / local shops ignored);
* proceeds are capped at the TCGplayer market price (never assume you sell above).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ── marketplace cost assumptions (TCGplayer seller, 2026) ───────────────────────
FEE_PCT = 0.1075 + 0.025      # marketplace commission + payment processing
FEE_FIXED = 0.30              # per order
TICK = 0.01                   # undercut increment


def ship_cost(landed_price: float) -> float:
    """Seller's own shipping cost by value tier (envelope → tracked → insured)."""
    if landed_price < 20:
        return 0.75
    if landed_price < 50:
        return 1.50
    if landed_price < 250:
        return 5.50
    return 12.00


def net_proceeds(landed_price: float) -> float:
    """Cash kept per unit sold at a landed price (seller charges shipping = cost)."""
    return max(0.0, landed_price * (1 - FEE_PCT) - FEE_FIXED - ship_cost(landed_price))


# ── layer 1: demand ─────────────────────────────────────────────────────────────

@dataclass
class Demand:
    lam_day: float      # units / day (recent window)
    phi: float          # dispersion index of weekly counts (≥ 1)

    def quantile(self, h_days: float, alpha: float) -> int:
        """α-quantile of units demanded in h days (α small = stressed/low demand)."""
        m = self.lam_day * h_days
        if m <= 0:
            return 0
        if self.phi <= 1.0 + 1e-9:
            return int(stats.poisson.ppf(alpha, m))
        n = m / (self.phi - 1.0)          # NegBin(n, p): mean m, var φm
        return int(stats.nbinom.ppf(alpha, n, 1.0 / self.phi))


def fit_demand(weekly_qty: list[float], recent_weeks: int = 13) -> Demand:
    q = np.asarray([x for x in weekly_qty if x is not None and not math.isnan(x)], float)
    if q.size == 0:
        return Demand(0.0, 1.0)
    recent = q[-recent_weeks:]
    mean_all = q.mean()
    phi = float(q.var(ddof=1) / mean_all) if q.size > 2 and mean_all > 0 else 1.0
    return Demand(lam_day=float(recent.mean() / 7.0), phi=max(1.0, phi))


# ── layer 2: queue liquidation ──────────────────────────────────────────────────

@dataclass
class Level:
    landed: float       # price + shipping
    qty: int


def ladder_from_listings(rows: list[dict]) -> list[Level]:
    """Aggregate raw listings into price levels sorted by landed price."""
    agg: dict[float, int] = {}
    for r in rows:
        p = round(float(r["price"]) + float(r.get("shipping", 0.0)), 2)
        agg[p] = agg.get(p, 0) + int(r["qty"])
    return [Level(p, q) for p, q in sorted(agg.items())]


def clean_ladder(ladder: list[Level], market: float, floor: float = 0.60) -> list[Level]:
    """Drop asks below floor·market.  They are stale/mislabelled listings (a real
    ask at 10% of market would have been bought) and would otherwise force the
    model to 'undercut' a price nobody can actually compete with."""
    return [l for l in ladder if l.landed >= floor * market]


def scale_ladder(ladder: list[Level], price_mult: float = 1.0, depth_mult: float = 1.0) -> list[Level]:
    return [Level(round(l.landed * price_mult, 2), max(1, int(round(l.qty * depth_mult))))
            for l in ladder]


@dataclass
class Liquidation:
    h_days: float
    alpha: float
    q: int
    demand: int              # D_h,α
    price: float             # landed price p* for the units sold on-market
    sold_market: int
    dumped: int
    proceeds: float          # total cash
    haircut: float           # 1 − proceeds / (q·market)


def liquidate(ladder: list[Level], market: float, q: int, demand: Demand, h_days: float,
              alpha: float = 0.05, dump_frac: float = 0.60) -> Liquidation:
    """Best (highest) price that clears q units within h days at confidence 1−α."""
    D = demand.quantile(h_days, alpha)
    cap = market                      # never assume a sale above market
    if D >= q:
        room = D - q                  # units we may tolerate ahead of us
        price = cap
        ahead = 0
        for lvl in ladder:            # undercut the first level that would overflow room
            if ahead + lvl.qty > room:
                price = min(cap, lvl.landed - TICK)
                break
            ahead += lvl.qty
        sold, dumped = q, 0
    else:                             # too few buyers: undercut the cheapest ask, rest to the shop
        best = ladder[0].landed - TICK if ladder else cap
        price = min(cap, best)
        sold, dumped = D, q - D
    price = max(price, 0.0)
    bid = dump_frac * market          # outside option: sell to a shop instead
    if net_proceeds(price) < bid:     # undercutting that deep is worse than the shop bid
        sold, dumped = 0, q
    proceeds = sold * net_proceeds(price) + dumped * bid
    hc = 1.0 - proceeds / (q * market) if q > 0 and market > 0 else float("nan")
    return Liquidation(h_days, alpha, q, D, round(price, 2), sold, dumped, proceeds, hc)


def days_to_liquidate(ladder, market, q, demand, max_haircut: float, alpha: float = 0.05,
                      max_days: int = 365, dump_frac: float = 0.60) -> int | None:
    """Shortest horizon (days) whose stressed haircut ≤ max_haircut, else None."""
    for h in list(range(1, 31)) + list(range(35, max_days + 1, 5)):
        if liquidate(ladder, market, q, demand, h, alpha, dump_frac).haircut <= max_haircut:
            return h
    return None


# ── layer 3: exogenous LVaR (Bangia) ────────────────────────────────────────────

def cornish_fisher_var(r: np.ndarray, conf: float, horizon: int = 1) -> float:
    """Loss quantile over ``horizon`` periods (positive = loss, log-return units).

    Drift scales with H, volatility with √H.  Positive drift is not credited
    (Bangia et al. set E[r] = 0; crediting a hype-driven uptrend would shrink the
    risk number exactly when it is least trustworthy), negative drift is kept.
    The skew/kurtosis adjustment uses the per-period moments — under iid they
    would shrink as s/√H and k/H, so keeping them is the conservative choice.

    Cornish–Fisher is only valid while z ↦ z_cf is monotone.  Extreme moments
    (thinly traded cards: mostly zero weeks plus one jump) break that and can
    even flip the sign of the loss, so outside the valid domain the standardized
    empirical quantile is used instead, never less severe than the normal z.
    The loss is floored at 0.
    """
    r = r[~np.isnan(r)]
    if r.size < 8:
        return float("nan")
    z = stats.norm.ppf(1 - conf)
    s, k = stats.skew(r), stats.kurtosis(r)   # excess kurtosis
    sd = float(r.std(ddof=1))
    if cf_valid(s, k):
        zq = cf_z(z, s, k)
    else:
        z_emp = (float(np.quantile(r, 1 - conf)) - float(r.mean())) / sd if sd > 0 else z
        zq = min(z, z_emp)
    drift = min(float(r.mean()), 0.0) * horizon
    return max(0.0, float(-(drift + zq * sd * math.sqrt(horizon))))


def cf_z(z, s: float, k: float):
    """Cornish–Fisher adjusted standard-normal quantile."""
    return z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36


_ZGRID = np.linspace(-4.0, 4.0, 161)


def cf_valid(s: float, k: float) -> bool:
    """True if the Cornish–Fisher map is strictly increasing on z ∈ [−4, 4]."""
    return bool(np.all(np.diff(cf_z(_ZGRID, s, k)) > 0))


@dataclass
class Bangia:
    price_var: float          # % loss, horizon H weeks
    spread_mean: float
    spread_sd: float
    a: float                  # spread scaling factor actually used
    liq_addon: float
    lvar: float


def spread_scale(s: np.ndarray, conf: float) -> float:
    """Bangia's scaling factor a: μ_S + a·σ_S should cover the conf-tail of the
    spread.  Spreads are far from normal (the paper finds a ≈ 2.0–4.5), so a is
    read off the empirical quantile, floored at the normal z for thin samples."""
    z = float(stats.norm.ppf(conf))
    sd = float(s.std(ddof=1))
    if s.size < 20 or sd <= 0:
        return z
    return max(z, (float(np.quantile(s, conf)) - float(s.mean())) / sd)


def bangia_lvar(weekly_ret: np.ndarray, weekly_spread: np.ndarray, conf: float = 0.95,
                horizon_weeks: int = 4) -> Bangia:
    """Bangia et al. (1999):  LVaR/P = 1 − e^{r_α,H} + ½(μ_S + a·σ_S).

    The spread term is a one-off exit cost, so it does not scale with H.
    Additive form assumes the worst price and the worst spread coincide.
    """
    v = cornish_fisher_var(weekly_ret, conf, horizon_weeks)
    pv = 1 - math.exp(-v) if not math.isnan(v) else float("nan")
    s = weekly_spread[~np.isnan(weekly_spread)]
    if s.size > 2:
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        a = spread_scale(s, conf)
    else:
        mu = sd = a = float("nan")
    add = 0.5 * (mu + a * sd)
    return Bangia(pv, mu, sd, a, add, pv + add)


# ── layer 4: diagnostics ────────────────────────────────────────────────────────

def amihud(weekly_ret: np.ndarray, weekly_dollar_vol: np.ndarray) -> float:
    """Mean |r| per $1k traded (higher = each dollar moves price more)."""
    m = (weekly_dollar_vol > 0) & ~np.isnan(weekly_ret)
    if m.sum() < 4:
        return float("nan")
    return float(np.mean(np.abs(weekly_ret[m]) / weekly_dollar_vol[m]) * 1e3)


# ── per-position bundle ─────────────────────────────────────────────────────────

@dataclass
class PositionRisk:
    name: str
    qty: int
    market: float
    value: float
    lam_day: float
    phi: float
    book_depth: int                      # units listed (NM, English)
    days_of_supply: float                # book_depth / λ
    best_ask: float
    liq: dict = field(default_factory=dict)          # {(h, α): Liquidation}
    days_to_liq: int | None = None        # stressed haircut ≤ MAX_HAIRCUT
    bangia: Bangia | None = None
    amihud: float = float("nan")
    combined_30d: float = float("nan")   # 1 − (1−VaR_30d)(1−haircut_30d,5%)
    dump_frac: float = 0.60              # shop / buylist bid used for this position
    bid_source: str = "assumed"          # assumed | calibrated <bucket> | position
    game: str = "Pokémon"
    trade_weeks: int = 0                 # weeks with ≥ 1 sale (of 52); < MIN_TRADE_WEEKS → VaR unreliable
