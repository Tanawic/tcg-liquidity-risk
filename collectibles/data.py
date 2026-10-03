"""Local data loader.  The model never touches the network: every input is a file.

Layout of a dataset directory (see data/sample/ and README "Use your own data"):

    portfolio.json            {"positions": [{"id", "name", "qty", "reprint_risk", ["dump_frac"]}]}
    weekly/<id>.csv           date, market, qty, txn, low, high      (one row per week, any order)
    listings/<id>.csv         price, shipping, qty                   (current asks, one row per listing)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_weekly(data_dir: Path, item_id: str) -> pd.DataFrame:
    """Weekly history with the derived columns the model uses: ret, spread, dvol."""
    df = pd.read_csv(Path(data_dir) / "weekly" / f"{item_id}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ("market", "qty", "txn", "low", "high"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["market"] <= 0, "market"] = np.nan
    df["market"] = df["market"].ffill()
    df["ret"] = np.log(df["market"]).diff()
    ok = (df["txn"] >= 2) & (df["market"] > 0)
    df["spread"] = np.where(ok, (df["high"] - df["low"]) / df["market"], np.nan)
    df["dvol"] = df["qty"].fillna(0) * df["market"]
    return df


def load_listings(data_dir: Path, item_id: str) -> list[dict]:
    df = pd.read_csv(Path(data_dir) / "listings" / f"{item_id}.csv")
    return [{"price": float(r.price), "shipping": float(r.shipping or 0.0), "qty": int(r.qty)}
            for r in df.itertuples()]
