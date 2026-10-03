# TCG Liquidity Risk Model

**Market price is not the price you can sell at.**

A liquidity-risk model for trading-card portfolios. It answers one question:
*if I had to sell, how much cash would I actually get, and how long would I have to wait?*

> The repository ships a **synthetic** sample dataset and **never downloads market data**. Every number it
> produces out of the box comes from made-up data. Bring your own data to use it for real (see below).

---

## The problem

Collectors value their portfolios at "market price". But on card marketplaces there is **no bid side**.
Nobody posts a standing offer to buy. To sell, you join a queue of other sellers and wait for a buyer to arrive.

So the real exit price depends on things the market price ignores:

- how many other sellers are already listed below you,
- how many buyers will show up before your deadline,
- fees, shipping, and the discount a shop charges if you can't wait.

That gap between *what it's worth* and *what you can get, by when* is liquidity risk.

## The inspiration

What risk does a collector or seller carry just by holding cards? And why do buyers, shops especially,
only pay below market? Because once they buy, they carry the same risk: they will have to sell into the same
thin market later. The discount is the buyer pricing in the liquidity risk they take off your hands.

This is the search-and-bargaining story of over-the-counter markets
(Duffie, Gârleanu & Pedersen 2007; Weill 2020). A card marketplace has the same structure, with one advantage:
listings and sales are visible, so the theory can be measured.

## How it works

| Layer | What it does | Method |
|---|---|---|
| 1. Demand | Forecast how many units will sell in *h* days | Weekly sales → rate λ and dispersion φ; D<sub>h</sub> ~ Negative Binomial (mean λh, var φλh). Real sales are clumpy, so Poisson overstates demand in quiet weeks. |
| 2. Queue | Find the price that sells *q* units within *h* days at confidence 1−α | Sit ahead of at most D<sub>h,α</sub> − q units of the ask ladder; undercut by one cent. Unsold units go to the outside option (shop bid). Net of fees and shipping → **haircut(q, h)**. |
| 3. Exogenous risk | Price risk + spread risk | Bangia et al. (1999) LVaR: Cornish–Fisher price VaR on weekly log returns + ½(μ<sub>S</sub> + a·σ<sub>S</sub>) of the sale-price range. |
| 4. Diagnostics | Quick liquidity reads | Days of supply (= listings ÷ daily sales), Amihud illiquidity. |
| Stress | What happens when hype ends | Demand falls **and** the queue thickens at the same time (more holders rush to list); reprint shock for in-print products. |

Combined loss at horizon *h*: `1 − (1 − VaR_h)(1 − haircut_h)`.

## Quickstart

```bash
pip install -r requirements.txt

python scripts/tcg_liquidity.py              # runs on the synthetic sample in data/sample/
python scripts/make_sample_data.py           # regenerate the sample (deterministic)
```

Output: `outputs/tcg_liquidity/latest_sample.md` (portfolio + per-holding tables), `latest_sample.json`, and
`haircut_curves_sample.png` (price–time frontier per holding and portfolio value under each scenario).

## Use your own data

The model only reads files. Create a folder with this layout:

```
my_data/
  portfolio.json
  weekly/<id>.csv      date, market, qty, txn, low, high    # one row per week (52 weeks is ideal)
  listings/<id>.csv    price, shipping, qty                 # every ask currently listed
```

```json
{
  "positions": [
    {"id": "my_card", "name": "My card", "qty": 1, "reprint_risk": false}
  ]
}
```

| Column | Meaning |
|---|---|
| `market` | the week's market price (e.g. average sale price) |
| `qty` | units sold that week |
| `txn` | number of orders that week |
| `low`, `high` | lowest / highest sale price that week (blank if no sales) |
| `price`, `shipping`, `qty` (listings) | one row per listing |

Optional per position: `dump_frac` (your shop/buylist offer as a share of market, default 0.60).

Then run `python scripts/tcg_liquidity.py --portfolio my_data/portfolio.json`.
How you collect the data is up to you; record it yourself or export it from a source whose terms allow it.

## Learn it step by step

- `notebooks/tcg_liquidity_walkthrough.ipynb`: the whole model rebuilt from scratch, one concept per step
  (concept → formula → code), runs offline on the sample data.
- `docs/TCG_Study_Checklist.pdf` (Thai): a study plan from the math fundamentals to the full model.

## Project structure

```
collectibles/
  data.py          local file loader (the only input path)
  liquidity.py     demand (NegBin), queue liquidation, Bangia LVaR, diagnostics
  portfolio.py     per-position and portfolio risk, stress scenarios
scripts/
  tcg_liquidity.py       CLI report
  make_sample_data.py    synthetic sample generator
data/sample/       synthetic dataset
notebooks/         step-by-step walkthrough
docs/              study checklist (Thai)
```

## Limitations

- The listing queue is treated as static over the horizon (no new sellers arrive). The stress scenarios thicken it to compensate.
- Only one marketplace's demand is counted.
- Proceeds are capped at the market price.
- The shop bid is an assumption (60% of market unless you set `dump_frac`).
- Stress scenarios are hypothetical, not calibrated to past events.
- The model says nothing about where card prices will go. It measures what an exit costs today.

## Disclaimer

Research and education only, not investment advice. The sample data is synthetic. Not affiliated with any
marketplace or trading-card publisher.

## References

- Duffie, D., Gârleanu, N., & Pedersen, L. H. (2007). Valuation in Over-the-Counter Markets. *Review of Financial Studies*, 20(6), 1865–1900.
- Weill, P.-O. (2020). The Search Theory of Over-the-Counter Markets. *Annual Review of Economics*, 12, 747–773.
- Bangia, A., Diebold, F. X., Schuermann, T., & Stroughair, J. D. (1999). Modeling Liquidity Risk, with Implications for Traditional Market Risk Measurement and Management. Wharton Financial Institutions Center WP 99-06.
- Amihud, Y. (2002). Illiquidity and Stock Returns: Cross-Section and Time-Series Effects. *Journal of Financial Markets*, 5(1), 31–56.
