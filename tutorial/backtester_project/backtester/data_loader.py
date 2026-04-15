from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .datamodel import OrderDepth, Trade


@dataclass
class Snapshot:
    timestamp: int
    order_depths: Dict[str, OrderDepth]
    mids: Dict[str, float]


class HistoricalData:
    def __init__(self, price_path: str | Path, trade_path: str | Path, max_rows: int | None = None):
        self.price_path = Path(price_path)
        self.trade_path = Path(trade_path)
        self.prices = pd.read_csv(self.price_path, sep=";")
        self.trades = pd.read_csv(self.trade_path, sep=";")

        if max_rows is not None:
            self.prices = self.prices.iloc[:2*max_rows]
            self.trades = self.trades.iloc[:2*max_rows]

        self.prices = self.prices.sort_values(
            ["timestamp", "product"]
        ).reset_index(drop=True)
        self.trades = self.trades.sort_values(
            ["timestamp", "symbol"]
        ).reset_index(drop=True)

        self.products = sorted(self.prices["product"].unique().tolist())
        self.timestamps = sorted(self.prices["timestamp"].unique().tolist())

        self._trade_buckets: Dict[Tuple[int, int], Dict[str, List[Trade]]] = {}
        self._build_trade_buckets()

    def _build_trade_buckets(self) -> None:
        # Bucket trades into half-open intervals [t_i, t_{i+1}) for passive-fill simulation.
        timestamps = self.timestamps
        if len(timestamps) < 2:
            return

        for i in range(len(timestamps) - 1):
            t0, t1 = timestamps[i], timestamps[i + 1]
            mask = (self.trades["timestamp"] >= t0) & (
                self.trades["timestamp"] < t1
            )
            df = self.trades.loc[mask]
            bucket: Dict[str, List[Trade]] = {}
            for symbol, g in df.groupby("symbol"):
                bucket[symbol] = [
                    Trade(
                        symbol=symbol,
                        price=int(row.price),
                        quantity=int(row.quantity),
                        buyer=None if pd.isna(row.buyer) else str(row.buyer),
                        seller=(
                            None if pd.isna(row.seller) else str(row.seller)
                        ),
                        timestamp=int(row.timestamp),
                    )
                    for row in g.itertuples(index=False)
                ]
            self._trade_buckets[(t0, t1)] = bucket

    def get_interval_trades(
        self, start_ts: int, end_ts: int
    ) -> Dict[str, List[Trade]]:
        return self._trade_buckets.get((start_ts, end_ts), {})

    def iter_snapshots(self) -> Iterable[Snapshot]:
        for ts, g in self.prices.groupby("timestamp", sort=True):
            order_depths: Dict[str, OrderDepth] = {}
            mids: Dict[str, float] = {}
            for row in g.itertuples(index=False):
                buy_orders = {}
                sell_orders = {}
                for level in (1, 2, 3):
                    bp = getattr(row, f"bid_price_{level}")
                    bv = getattr(row, f"bid_volume_{level}")
                    ap = getattr(row, f"ask_price_{level}")
                    av = getattr(row, f"ask_volume_{level}")

                    if pd.notna(bp) and pd.notna(bv):
                        buy_orders[int(bp)] = int(bv)
                    if pd.notna(ap) and pd.notna(av):
                        sell_orders[int(ap)] = -int(av)

                od = OrderDepth()
                od.buy_orders = buy_orders
                od.sell_orders = sell_orders
                order_depths[str(row.product)] = od
                mids[str(row.product)] = float(row.mid_price)

            yield Snapshot(
                timestamp=int(ts), order_depths=order_depths, mids=mids
            )
