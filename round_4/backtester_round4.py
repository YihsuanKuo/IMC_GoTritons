#!/usr/bin/env python3
"""
Lightweight IMC Prosperity Round 4 backtester.

Put this file inside your `round_4/` folder, next to `insider_trade.py` and the
`data/` folder, then run for example:

    python3 backtester_round4.py --strategy insider_trade.py --data-dir data --days 1 2 3

Useful modes:
    --passive-fill none    only fills orders that cross the visible book
    --passive-fill touch   also approximates passive fills using public trades

This is an approximation of the official simulator. It is mainly useful for
relative comparisons / ablation tests, not for exact leaderboard PnL.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Fallback datamodel
# ---------------------------------------------------------------------------


def ensure_datamodel(strategy_dir: Path):
    """Import the user's datamodel if available; otherwise inject a minimal one."""
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    try:
        import datamodel  # type: ignore

        return datamodel
    except Exception:
        pass

    dm = types.ModuleType("datamodel")

    class Order:
        def __init__(self, symbol: str, price: int, quantity: int):
            self.symbol = symbol
            self.price = int(price)
            self.quantity = int(quantity)

        def __repr__(self):
            return f"Order({self.symbol}, {self.price}, {self.quantity})"

    class OrderDepth:
        def __init__(self):
            self.buy_orders: Dict[int, int] = {}
            self.sell_orders: Dict[int, int] = {}

    class Trade:
        def __init__(
            self,
            symbol: str,
            price: int,
            quantity: int,
            buyer: Optional[str] = None,
            seller: Optional[str] = None,
            timestamp: int = 0,
        ):
            self.symbol = symbol
            self.price = int(price)
            self.quantity = int(quantity)
            self.buyer = buyer
            self.seller = seller
            self.timestamp = int(timestamp)

        def __repr__(self):
            return (
                f"Trade({self.symbol}, {self.price}, {self.quantity}, "
                f"{self.buyer}, {self.seller}, {self.timestamp})"
            )

    class Listing(dict):
        def __init__(self, symbol: str, product: str, denomination: str):
            super().__init__(
                symbol=symbol, product=product, denomination=denomination
            )
            self.symbol = symbol
            self.product = product
            self.denomination = denomination

    class Observation:
        def __init__(
            self, plainValueObservations=None, conversionObservations=None
        ):
            self.plainValueObservations = plainValueObservations or {}
            self.conversionObservations = conversionObservations or {}

    class TradingState:
        def __init__(
            self,
            traderData: str,
            timestamp: int,
            listings: Dict[str, Listing],
            order_depths: Dict[str, OrderDepth],
            own_trades: Dict[str, List[Trade]],
            market_trades: Dict[str, List[Trade]],
            position: Dict[str, int],
            observations: Optional[Observation] = None,
        ):
            self.traderData = traderData
            self.timestamp = int(timestamp)
            self.listings = listings
            self.order_depths = order_depths
            self.own_trades = own_trades
            self.market_trades = market_trades
            self.position = position
            self.observations = observations or Observation()

    dm.Order = Order
    dm.OrderDepth = OrderDepth
    dm.Trade = Trade
    dm.Listing = Listing
    dm.Observation = Observation
    dm.TradingState = TradingState
    sys.modules["datamodel"] = dm
    return dm


# ---------------------------------------------------------------------------
# Loading strategy and data
# ---------------------------------------------------------------------------


def load_strategy(strategy_path: Path):
    dm = ensure_datamodel(strategy_path.parent)
    spec = importlib.util.spec_from_file_location(
        "user_strategy", strategy_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import strategy from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_strategy"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Trader"):
        raise RuntimeError("Strategy file does not define class Trader")
    return module.Trader(), dm


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_day_files(
    data_dir: Path, days: Iterable[int]
) -> Tuple[List[Path], List[Path]]:
    price_files = []
    trade_files = []
    for day in days:
        p = data_dir / f"prices_round_4_day_{day}.csv"
        t = data_dir / f"trades_round_4_day_{day}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing price file: {p}")
        if not t.exists():
            raise FileNotFoundError(f"Missing trade file: {t}")
        price_files.append(p)
        trade_files.append(t)
    return price_files, trade_files


def normalize_price_df(df: pd.DataFrame, day: int) -> pd.DataFrame:
    df = df.copy()
    if "day" not in df.columns:
        df["day"] = day
    if "product" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "product"})
    df["product"] = df["product"].astype(str).str.strip()
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def normalize_trade_df(df: pd.DataFrame, day: int) -> pd.DataFrame:
    df = df.copy()
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "day",
                "timestamp",
                "symbol",
                "price",
                "quantity",
                "buyer",
                "seller",
            ]
        )
    if "day" not in df.columns:
        df["day"] = day
    if "symbol" not in df.columns and "product" in df.columns:
        df = df.rename(columns={"product": "symbol"})
    for c in ["buyer", "seller"]:
        if c not in df.columns:
            df[c] = None
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["timestamp"] = df["timestamp"].astype(int)
    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    return df


def load_round_data(data_dir: Path, days: List[int]):
    price_files, trade_files = find_day_files(data_dir, days)
    prices = []
    trades = []
    for day, p, t in zip(days, price_files, trade_files):
        prices.append(normalize_price_df(read_csv_auto(p), day))
        trades.append(normalize_trade_df(read_csv_auto(t), day))
    prices_df = pd.concat(prices, ignore_index=True).sort_values(
        ["day", "timestamp", "product"]
    )
    trades_df = pd.concat(trades, ignore_index=True).sort_values(
        ["day", "timestamp", "symbol"]
    )
    return prices_df, trades_df


# ---------------------------------------------------------------------------
# State construction helpers
# ---------------------------------------------------------------------------


def finite_number(x: Any) -> bool:
    try:
        return (
            x is not None
            and not (isinstance(x, float) and math.isnan(x))
            and not pd.isna(x)
        )
    except Exception:
        return False


def row_to_depth(row: pd.Series, dm) -> Any:
    od = dm.OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if finite_number(bp) and finite_number(bv) and int(bv) != 0:
            od.buy_orders[int(bp)] = int(bv)
        if finite_number(ap) and finite_number(av) and int(av) != 0:
            # IMC datamodel convention: sell quantities are negative.
            od.sell_orders[int(ap)] = -abs(int(av))
    return od


def make_trade(
    dm, symbol: str, price: int, qty: int, buyer: Any, seller: Any, ts: int
):
    try:
        return dm.Trade(symbol, int(price), int(qty), buyer, seller, int(ts))
    except TypeError:
        tr = dm.Trade(symbol, int(price), int(qty))
        tr.buyer = buyer
        tr.seller = seller
        tr.timestamp = int(ts)
        return tr


def build_market_trades(
    trades_slice: pd.DataFrame, dm
) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = defaultdict(list)
    if trades_slice is None or trades_slice.empty:
        return dict(out)
    for _, r in trades_slice.iterrows():
        symbol = str(r["symbol"]).strip()
        out[symbol].append(
            make_trade(
                dm,
                symbol,
                int(r["price"]),
                int(r["quantity"]),
                None if pd.isna(r.get("buyer")) else r.get("buyer"),
                None if pd.isna(r.get("seller")) else r.get("seller"),
                int(r["timestamp"]),
            )
        )
    return dict(out)


def make_listing(dm, product: str):
    try:
        return dm.Listing(product, product, "SEASHELLS")
    except Exception:
        return {
            "symbol": product,
            "product": product,
            "denomination": "SEASHELLS",
        }


def make_observation(dm):
    try:
        return dm.Observation({}, {})
    except Exception:
        try:
            return dm.Observation()
        except Exception:
            return None


def make_state(
    dm,
    trader_data: str,
    timestamp: int,
    products: List[str],
    order_depths: Dict[str, Any],
    own_trades: Dict[str, List[Any]],
    market_trades: Dict[str, List[Any]],
    position: Dict[str, int],
):
    listings = {p: make_listing(dm, p) for p in products}
    obs = make_observation(dm)
    try:
        return dm.TradingState(
            trader_data,
            int(timestamp),
            listings,
            order_depths,
            own_trades,
            market_trades,
            dict(position),
            obs,
        )
    except TypeError:
        # Older datamodel without traderData.
        return dm.TradingState(
            int(timestamp),
            listings,
            order_depths,
            own_trades,
            market_trades,
            dict(position),
            obs,
        )


def normalize_result(raw_result) -> Tuple[Dict[str, List[Any]], int, str]:
    if isinstance(raw_result, tuple):
        if len(raw_result) == 3:
            result, conversions, trader_data = raw_result
        elif len(raw_result) == 2:
            result, conversions = raw_result
            trader_data = ""
        else:
            result = raw_result[0]
            conversions = 0
            trader_data = ""
    else:
        result = raw_result
        conversions = 0
        trader_data = ""
    if result is None:
        result = {}
    if trader_data is None:
        trader_data = ""
    return result, int(conversions or 0), str(trader_data)


# ---------------------------------------------------------------------------
# Execution model
# ---------------------------------------------------------------------------


def copy_depth(od) -> Tuple[Dict[int, int], Dict[int, int]]:
    return dict(od.buy_orders), dict(od.sell_orders)


def crossed_fill_orders(
    orders: Dict[str, List[Any]],
    depths: Dict[str, Any],
    dm,
    timestamp: int,
    position: Dict[str, int],
    cash_by_product: Dict[str, float],
    fill_log: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    own_trades: Dict[str, List[Any]] = defaultdict(list)

    for product, product_orders in (orders or {}).items():
        if product not in depths:
            continue
        buy_book, sell_book = copy_depth(depths[product])

        for order in product_orders:
            qty = int(getattr(order, "quantity", 0))
            limit_px = int(getattr(order, "price", 0))
            if qty == 0:
                continue

            remaining = abs(qty)
            signed_side = 1 if qty > 0 else -1

            if signed_side > 0:
                # Buy: execute against asks at ask <= limit price.
                for ask_px in sorted(list(sell_book.keys())):
                    if remaining <= 0 or ask_px > limit_px:
                        break
                    avail = -int(sell_book[ask_px])
                    if avail <= 0:
                        continue
                    fill = min(remaining, avail)
                    remaining -= fill
                    sell_book[ask_px] += fill
                    position[product] = position.get(product, 0) + fill
                    cash_by_product[product] -= fill * ask_px
                    tr = make_trade(
                        dm,
                        product,
                        ask_px,
                        fill,
                        "SUBMISSION",
                        "MARKET",
                        timestamp,
                    )
                    own_trades[product].append(tr)
                    fill_log.append(
                        dict(
                            timestamp=timestamp,
                            product=product,
                            side="BUY",
                            price=ask_px,
                            quantity=fill,
                            mode="cross",
                        )
                    )
            else:
                # Sell: execute against bids at bid >= limit price.
                for bid_px in sorted(list(buy_book.keys()), reverse=True):
                    if remaining <= 0 or bid_px < limit_px:
                        break
                    avail = int(buy_book[bid_px])
                    if avail <= 0:
                        continue
                    fill = min(remaining, avail)
                    remaining -= fill
                    buy_book[bid_px] -= fill
                    position[product] = position.get(product, 0) - fill
                    cash_by_product[product] += fill * bid_px
                    tr = make_trade(
                        dm,
                        product,
                        bid_px,
                        -fill,
                        "MARKET",
                        "SUBMISSION",
                        timestamp,
                    )
                    own_trades[product].append(tr)
                    fill_log.append(
                        dict(
                            timestamp=timestamp,
                            product=product,
                            side="SELL",
                            price=bid_px,
                            quantity=fill,
                            mode="cross",
                        )
                    )

    return dict(own_trades)


def passive_touch_fills(
    orders: Dict[str, List[Any]],
    market_trades: Dict[str, List[Any]],
    depths: Dict[str, Any],
    dm,
    timestamp: int,
    position: Dict[str, int],
    cash_by_product: Dict[str, float],
    fill_log: List[Dict[str, Any]],
    already_filled: Dict[str, List[Any]],
    queue_fraction: float = 0.35,
) -> Dict[str, List[Any]]:
    """Approximate passive fills from public trades.

    If a public print occurs through our limit price, assume we receive up to
    `queue_fraction` of the touched volume. This is deliberately conservative.
    """
    own_trades: Dict[str, List[Any]] = defaultdict(list)
    for k, v in already_filled.items():
        own_trades[k].extend(v)

    for product, product_orders in (orders or {}).items():
        if product not in market_trades or product not in depths:
            continue

        # Aggregate visible public trade volume by price.
        buy_touch_vol = defaultdict(
            int
        )  # for our passive sells: prints >= ask quote
        sell_touch_vol = defaultdict(
            int
        )  # for our passive buys: prints <= bid quote
        for tr in market_trades.get(product, []):
            px = int(getattr(tr, "price", 0))
            q = abs(int(getattr(tr, "quantity", 0) or 0))
            buyer = getattr(tr, "buyer", None)
            seller = getattr(tr, "seller", None)
            # If direction is unknown, count it for both touch checks.
            if buyer is not None:
                buy_touch_vol[px] += q
            if seller is not None:
                sell_touch_vol[px] += q
            if buyer is None and seller is None:
                buy_touch_vol[px] += q
                sell_touch_vol[px] += q

        od = depths[product]
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        for order in product_orders:
            qty = int(getattr(order, "quantity", 0))
            px = int(getattr(order, "price", 0))
            if qty == 0:
                continue

            # Skip orders that would already have crossed; those were handled above.
            if qty > 0 and best_ask is not None and px >= best_ask:
                continue
            if qty < 0 and best_bid is not None and px <= best_bid:
                continue

            if qty > 0:
                touched = sum(q for p, q in sell_touch_vol.items() if p <= px)
                fill = min(abs(qty), int(touched * queue_fraction))
                if fill > 0:
                    position[product] = position.get(product, 0) + fill
                    cash_by_product[product] -= fill * px
                    tr = make_trade(
                        dm,
                        product,
                        px,
                        fill,
                        "SUBMISSION",
                        "MARKET",
                        timestamp,
                    )
                    own_trades[product].append(tr)
                    fill_log.append(
                        dict(
                            timestamp=timestamp,
                            product=product,
                            side="BUY",
                            price=px,
                            quantity=fill,
                            mode="passive",
                        )
                    )
            else:
                touched = sum(q for p, q in buy_touch_vol.items() if p >= px)
                fill = min(abs(qty), int(touched * queue_fraction))
                if fill > 0:
                    position[product] = position.get(product, 0) - fill
                    cash_by_product[product] += fill * px
                    tr = make_trade(
                        dm,
                        product,
                        px,
                        -fill,
                        "MARKET",
                        "SUBMISSION",
                        timestamp,
                    )
                    own_trades[product].append(tr)
                    fill_log.append(
                        dict(
                            timestamp=timestamp,
                            product=product,
                            side="SELL",
                            price=px,
                            quantity=fill,
                            mode="passive",
                        )
                    )

    return dict(own_trades)


def get_depth_mid(od) -> Optional[float]:
    if od is None:
        return None
    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
    if od.buy_orders:
        return float(max(od.buy_orders.keys()))
    if od.sell_orders:
        return float(min(od.sell_orders.keys()))
    return None


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------


def run_backtest(
    strategy_path: Path,
    data_dir: Path,
    days: List[int],
    passive_fill: str = "touch",
    reset_each_day: bool = True,
    queue_fraction: float = 0.35,
    output_dir: Optional[Path] = None,
):
    trader, dm = load_strategy(strategy_path)
    prices_df, trades_df = load_round_data(data_dir, days)

    products = sorted(prices_df["product"].unique().tolist())
    position: Dict[str, int] = defaultdict(int)
    cash_by_product: Dict[str, float] = defaultdict(float)
    trader_data = ""
    last_mid: Dict[str, float] = {}
    last_own_trades: Dict[str, List[Any]] = {}

    pnl_rows: List[Dict[str, Any]] = []
    fill_log: List[Dict[str, Any]] = []

    for day in days:
        day_prices = prices_df[prices_df["day"] == day]
        day_trades = trades_df[trades_df["day"] == day]

        if reset_each_day:
            position = defaultdict(int)
            cash_by_product = defaultdict(float)
            trader_data = ""
            last_mid = {}
            last_own_trades = {}
            # Recreate the trader so internal state resets between days.
            trader, dm = load_strategy(strategy_path)

        for ts, ts_prices in day_prices.groupby("timestamp", sort=True):
            order_depths = {
                str(r["product"]): row_to_depth(r, dm)
                for _, r in ts_prices.iterrows()
            }

            for p, od in order_depths.items():
                mid = get_depth_mid(od)
                if mid is not None and mid != 0:
                    last_mid[p] = mid

            market_slice = day_trades[day_trades["timestamp"] == ts]
            market_trades = build_market_trades(market_slice, dm)

            state = make_state(
                dm,
                trader_data,
                int(ts),
                products,
                order_depths,
                last_own_trades,
                market_trades,
                dict(position),
            )

            try:
                raw_result = trader.run(state)
            except Exception as e:
                raise RuntimeError(
                    f"Strategy crashed at day={day}, timestamp={ts}: {e}"
                ) from e

            result, _conversions, trader_data = normalize_result(raw_result)

            crossed = crossed_fill_orders(
                result,
                order_depths,
                dm,
                int(ts),
                position,
                cash_by_product,
                fill_log,
            )
            if passive_fill == "touch":
                last_own_trades = passive_touch_fills(
                    result,
                    market_trades,
                    order_depths,
                    dm,
                    int(ts),
                    position,
                    cash_by_product,
                    fill_log,
                    crossed,
                    queue_fraction=queue_fraction,
                )
            else:
                last_own_trades = crossed

            mtm_by_product = {}
            total_pnl = 0.0
            for p in products:
                mid = last_mid.get(p)
                cash = cash_by_product.get(p, 0.0)
                pos = position.get(p, 0)
                pnl = cash + (pos * mid if mid is not None else 0.0)
                mtm_by_product[p] = pnl
                total_pnl += pnl

            row = {
                "day": day,
                "timestamp": int(ts),
                "total_pnl": total_pnl,
                "total_abs_pos": sum(
                    abs(position.get(p, 0)) for p in products
                ),
            }
            for p in products:
                row[f"pos_{p}"] = position.get(p, 0)
                row[f"pnl_{p}"] = mtm_by_product[p]
            pnl_rows.append(row)

    pnl_df = pd.DataFrame(pnl_rows)
    fills_df = pd.DataFrame(fill_log)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pnl_df.to_csv(output_dir / "backtest_pnl.csv", index=False)
        fills_df.to_csv(output_dir / "backtest_fills.csv", index=False)

    return pnl_df, fills_df


def summarize(pnl_df: pd.DataFrame, fills_df: pd.DataFrame):
    if pnl_df.empty:
        print("No PnL rows generated.")
        return
    last = pnl_df.iloc[-1]
    print("\n========== BACKTEST SUMMARY ==========")
    print(f"Final total PnL: {last['total_pnl']:.2f}")
    print(f"Final total abs position: {int(last['total_abs_pos'])}")
    print(f"Number of fills: {0 if fills_df.empty else len(fills_df)}")

    pnl_cols = [c for c in pnl_df.columns if c.startswith("pnl_")]
    rows = []
    for c in pnl_cols:
        p = c[4:]
        pos_col = f"pos_{p}"
        rows.append(
            (
                p,
                float(last[c]),
                int(last[pos_col]) if pos_col in pnl_df.columns else 0,
            )
        )
    rows.sort(key=lambda x: x[1], reverse=True)
    print("\nPer-product final PnL / position:")
    for p, pnl, pos in rows:
        if abs(pnl) > 1e-9 or pos != 0:
            print(f"  {p:28s} pnl={pnl:10.2f}  pos={pos:5d}")

    if not fills_df.empty:
        print("\nFill counts by product:")
        counts = (
            fills_df.groupby(["product", "side", "mode"])
            .size()
            .reset_index(name="n")
        )
        print(counts.to_string(index=False))


def maybe_plot(pnl_df: pd.DataFrame, output_dir: Optional[Path]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot.")
        return
    if pnl_df.empty:
        return
    x = range(len(pnl_df))
    plt.figure(figsize=(12, 5))
    plt.plot(x, pnl_df["total_pnl"])
    plt.title("Backtest Total PnL")
    plt.xlabel("step")
    plt.ylabel("PnL")
    plt.grid(True, alpha=0.3)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "backtest_pnl.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default="insider_trade.py",
        help="Path to strategy .py file",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Path to round_4/data folder"
    )
    parser.add_argument("--days", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--passive-fill", choices=["none", "touch"], default="touch"
    )
    parser.add_argument("--queue-fraction", type=float, default=0.35)
    parser.add_argument("--no-reset-each-day", action="store_true")
    parser.add_argument("--output-dir", default="backtest_output")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    strategy_path = Path(args.strategy).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else None
    )

    pnl_df, fills_df = run_backtest(
        strategy_path=strategy_path,
        data_dir=data_dir,
        days=args.days,
        passive_fill=args.passive_fill,
        reset_each_day=not args.no_reset_each_day,
        queue_fraction=args.queue_fraction,
        output_dir=output_dir,
    )
    summarize(pnl_df, fills_df)
    if args.plot:
        maybe_plot(pnl_df, output_dir)


if __name__ == "__main__":
    main()
