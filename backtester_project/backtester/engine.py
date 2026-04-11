from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

from .datamodel import Order, OrderDepth, Trade, TradingState
from .data_loader import HistoricalData, Snapshot


@dataclass
class Fill:
    timestamp: int
    symbol: str
    side: str
    price: int
    quantity: int
    reason: str


@dataclass
class BacktestResult:
    fills: List[Fill] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_positions: Dict[str, int] = field(default_factory=dict)
    final_cash: float = 0.0
    final_pnl: float = 0.0


class ReplayBacktester:
    def __init__(
        self,
        data: HistoricalData,
        trader,
        position_limits: Dict[str, int] | None = None,
    ):
        self.data = data
        self.trader = trader
        self.position_limits = position_limits or getattr(
            trader, "POSITION_LIMITS", {}
        )

        self.positions: Dict[str, int] = defaultdict(int)
        self.cash: float = 0.0
        self.trader_data: str = ""
        self.own_trades: Dict[str, List[Trade]] = defaultdict(list)
        self.fills: List[Fill] = []
        self.equity_rows: List[dict] = []

    def _record_fill(
        self,
        timestamp: int,
        symbol: str,
        price: int,
        quantity: int,
        reason: str,
    ) -> None:
        side = "BUY" if quantity > 0 else "SELL"
        self.positions[symbol] += quantity
        self.cash -= price * quantity
        fill = Fill(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            price=price,
            quantity=abs(quantity),
            reason=reason,
        )
        self.fills.append(fill)
        self.own_trades[symbol].append(
            Trade(
                symbol=symbol,
                price=price,
                quantity=abs(quantity),
                buyer="SUBMISSION" if quantity > 0 else None,
                seller="SUBMISSION" if quantity < 0 else None,
                timestamp=timestamp,
            )
        )

    def _check_limits(self, symbol: str, proposed_delta: int) -> int:
        limit = self.position_limits.get(symbol)
        if limit is None:
            return proposed_delta
        new_pos = self.positions[symbol] + proposed_delta
        if new_pos > limit:
            return max(0, limit - self.positions[symbol])
        if new_pos < -limit:
            return min(0, -limit - self.positions[symbol])
        return proposed_delta

    def _execute_aggressive(
        self, ts: int, order: Order, depth: OrderDepth
    ) -> int:
        remaining = order.quantity
        if remaining > 0:
            asks = sorted(depth.sell_orders.items(), key=lambda x: x[0])
            for ask_price, ask_vol_neg in asks:
                ask_vol = -ask_vol_neg
                if remaining <= 0 or ask_price > order.price:
                    break
                fill_qty = min(remaining, ask_vol)
                clipped = self._check_limits(order.symbol, fill_qty)
                if clipped > 0:
                    self._record_fill(
                        ts, order.symbol, ask_price, clipped, "aggressive"
                    )
                    remaining -= clipped
        elif remaining < 0:
            bids = sorted(
                depth.buy_orders.items(), key=lambda x: x[0], reverse=True
            )
            need = -remaining
            for bid_price, bid_vol in bids:
                if need <= 0 or bid_price < order.price:
                    break
                fill_qty = min(need, bid_vol)
                clipped = -self._check_limits(order.symbol, -fill_qty)
                if clipped > 0:
                    self._record_fill(
                        ts, order.symbol, bid_price, -clipped, "aggressive"
                    )
                    need -= clipped
            remaining = -need
        return remaining

    def _execute_passive(
        self,
        start_ts: int,
        end_ts: int,
        order: Order,
        remaining: int,
        interval_trades: Dict[str, List[Trade]],
    ) -> None:
        if remaining == 0:
            return
        tape = interval_trades.get(order.symbol, [])
        if not tape:
            return

        if remaining > 0:
            cap = 0
            for tr in tape:
                if tr.price <= order.price:
                    cap += tr.quantity
            fill_qty = min(remaining, cap)
            clipped = self._check_limits(order.symbol, fill_qty)
            if clipped > 0:
                self._record_fill(
                    end_ts, order.symbol, order.price, clipped, "passive"
                )
        else:
            need = -remaining
            cap = 0
            for tr in tape:
                if tr.price >= order.price:
                    cap += tr.quantity
            fill_qty = min(need, cap)
            clipped = -self._check_limits(order.symbol, -fill_qty)
            if clipped > 0:
                self._record_fill(
                    end_ts, order.symbol, order.price, -clipped, "passive"
                )

    def _mark_to_market(self, ts: int, mids: Dict[str, float]) -> None:
        symbols = sorted(
            set(self.data.products)
            | set(self.positions.keys())
            | set(mids.keys())
        )

        inventory_value = sum(
            self.positions.get(s, 0) * mids.get(s, 0.0) for s in symbols
        )
        total = self.cash + inventory_value

        row = {
            "timestamp": ts,
            "cash": self.cash,
            "inventory_value": inventory_value,
            "total_pnl": total,
        }

        for symbol in symbols:
            row[f"pos_{symbol}"] = self.positions.get(symbol, 0)
            row[f"mid_{symbol}"] = mids.get(symbol)

        self.equity_rows.append(row)

    def run(self, max_steps: int | None = None) -> BacktestResult:
        snapshots = list(self.data.iter_snapshots())
        # if max_steps is not None:
        #     snapshots = snapshots[:max_steps]
        listings = {p: {"symbol": p} for p in self.data.products}

        for i, snap in enumerate(snapshots):
            next_ts = (
                snapshots[i + 1].timestamp
                if i + 1 < len(snapshots)
                else snap.timestamp + 100
            )
            interval_trades = self.data.get_interval_trades(
                snap.timestamp, next_ts
            )
            current_market_trades = {
                sym: list(trs) for sym, trs in interval_trades.items()
            }

            state = TradingState(
                traderData=self.trader_data,
                timestamp=snap.timestamp,
                listings=listings,
                order_depths=snap.order_depths,
                own_trades={k: list(v) for k, v in self.own_trades.items()},
                market_trades=current_market_trades,
                position=dict(self.positions),
                observations={},
            )

            result, conversions, new_trader_data = self.trader.run(state)
            _ = conversions  # not used in round-0 replay
            self.trader_data = new_trader_data or ""

            for symbol, orders in result.items():
                depth = snap.order_depths.get(symbol)
                if depth is None:
                    continue
                for order in orders:
                    if order.quantity == 0:
                        continue
                    remaining = self._execute_aggressive(
                        snap.timestamp, order, depth
                    )
                    self._execute_passive(
                        snap.timestamp,
                        next_ts,
                        order,
                        remaining,
                        interval_trades,
                    )

            self._mark_to_market(snap.timestamp, snap.mids)

        equity = pd.DataFrame(self.equity_rows)
        final_pnl = (
            float(equity["total_pnl"].iloc[-1]) if not equity.empty else 0.0
        )
        return BacktestResult(
            fills=self.fills,
            equity_curve=equity,
            final_positions=dict(self.positions),
            final_cash=self.cash,
            final_pnl=final_pnl,
        )
