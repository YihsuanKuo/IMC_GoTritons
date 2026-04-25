import argparse
import importlib.util
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    Trade,
    TradingState,
)

# =========================
# CONFIG
# =========================
ROUND_2_PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}


# =========================
# STRATEGY LOADING
# =========================
def load_trader_class(strategy_path: str):
    strategy_path = str(Path(strategy_path).resolve())
    spec = importlib.util.spec_from_file_location(
        "user_strategy", strategy_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load strategy module from {strategy_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Trader"):
        raise AttributeError("Strategy file must define a Trader class")

    return module.Trader


# =========================
# DATA LOADING
# =========================
def read_prices_with_day(csv_path: str, day: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    if "day" not in df.columns:
        df["day"] = day
    return df


def read_trades_with_day(csv_path: Optional[str], day: int) -> pd.DataFrame:
    if csv_path is None:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "buyer",
                "seller",
                "symbol",
                "currency",
                "price",
                "quantity",
                "day",
            ]
        )
    df = pd.read_csv(csv_path, sep=";")
    if "day" not in df.columns:
        df["day"] = day
    return df


def load_round2_data(
    data_dir: str,
    days: Optional[List[int]] = None,
    max_timestamp: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)

    price_files = {
        -1: data_dir / "prices_round_2_day_-1.csv",
        0: data_dir / "prices_round_2_day_0.csv",
        1: data_dir / "prices_round_2_day_1.csv",
    }

    trade_files = {
        -1: data_dir / "trades_round_2_day_-1.csv",
        0: data_dir / "trades_round_2_day_0.csv",
        1: data_dir / "trades_round_2_day_1.csv",
    }

    if days is None:
        days = [-1, 0, 1]

    prices = []
    trades = []

    for day in days:
        if day not in price_files:
            raise ValueError(f"Unsupported day: {day}")
        prices.append(read_prices_with_day(str(price_files[day]), day))
        trades.append(read_trades_with_day(str(trade_files[day]), day))

    prices_df = pd.concat(prices, ignore_index=True)
    trades_df = pd.concat(trades, ignore_index=True)

    if max_timestamp is not None:
        prices_df = prices_df[prices_df["timestamp"] <= max_timestamp].copy()
        if not trades_df.empty:
            trades_df = trades_df[
                trades_df["timestamp"] <= max_timestamp
            ].copy()

    prices_df = prices_df.sort_values(
        ["day", "timestamp", "product"]
    ).reset_index(drop=True)

    if not trades_df.empty:
        sort_cols = [
            c for c in ["day", "timestamp", "symbol"] if c in trades_df.columns
        ]
        trades_df = trades_df.sort_values(sort_cols).reset_index(drop=True)

    return prices_df, trades_df


# =========================
# ORDER BOOK BUILDING
# =========================
def build_order_depth(row: pd.Series) -> OrderDepth:
    od = OrderDepth()

    for level in [1, 2, 3]:
        bp = row.get(f"bid_price_{level}")
        bv = row.get(f"bid_volume_{level}")
        ap = row.get(f"ask_price_{level}")
        av = row.get(f"ask_volume_{level}")

        if pd.notna(bp) and pd.notna(bv):
            bp = int(bp)
            bv = int(bv)
            if bv > 0:
                od.buy_orders[bp] = bv

        if pd.notna(ap) and pd.notna(av):
            ap = int(ap)
            av = int(av)
            if av > 0:
                od.sell_orders[ap] = -av

    return od


def augment_order_depth_for_extra_access(
    order_depth: OrderDepth,
) -> OrderDepth:
    """
    Approximate +25% extra market access by increasing visible liquidity
    by ~25% at each visible price level.
    """
    new_od = OrderDepth()
    new_od.buy_orders = dict(order_depth.buy_orders)
    new_od.sell_orders = dict(order_depth.sell_orders)

    for price, vol in list(new_od.buy_orders.items()):
        extra = max(1, int(round(vol * 0.25)))
        new_od.buy_orders[price] = vol + extra

    for price, vol in list(new_od.sell_orders.items()):
        abs_vol = -vol
        extra = max(1, int(round(abs_vol * 0.25)))
        new_od.sell_orders[price] = -(abs_vol + extra)

    return new_od


def get_mark_price(
    row: pd.Series,
    od: OrderDepth,
    last_mid: Optional[float] = None,
) -> Optional[float]:
    """
    Mark-to-market price used for equity valuation.

    Priority:
    1. mid_price column from csv if valid and nonzero
    2. midpoint of best bid / best ask
    3. one-sided fallback (best bid or best ask)
    4. last known mid
    """
    csv_mid = row.get("mid_price", None)

    if pd.notna(csv_mid) and float(csv_mid) != 0:
        return float(csv_mid)

    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2

    if od.buy_orders:
        return float(max(od.buy_orders))

    if od.sell_orders:
        return float(min(od.sell_orders))

    return last_mid


# =========================
# LIMIT HANDLING
# =========================
def clip_order_to_limits(
    order: Order,
    product: str,
    current_pos: int,
    limits: Dict[str, int],
) -> int:
    limit = limits[product]
    qty = order.quantity

    if qty > 0:
        max_buy = limit - current_pos
        return max(0, min(qty, max_buy))
    else:
        max_sell = limit + current_pos
        allowed_sell = max(0, min(-qty, max_sell))
        return -allowed_sell


# =========================
# EXECUTION
# =========================
def execute_order_against_book(
    order: Order,
    product: str,
    order_depth: OrderDepth,
    current_pos: int,
    cash: float,
    timestamp: int,
) -> Tuple[List[Trade], int, float]:
    fills: List[Trade] = []
    pos = current_pos

    if order.quantity > 0:
        remaining = order.quantity
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if remaining <= 0:
                break
            if ask_price > order.price:
                break

            available = -order_depth.sell_orders[ask_price]
            fill_qty = min(remaining, available)
            if fill_qty <= 0:
                continue

            fills.append(
                Trade(
                    symbol=product,
                    price=ask_price,
                    quantity=fill_qty,
                    buyer="SUBMISSION",
                    seller="BOOK",
                    timestamp=timestamp,
                )
            )

            remaining -= fill_qty
            pos += fill_qty
            cash -= ask_price * fill_qty
            order_depth.sell_orders[ask_price] += fill_qty

            if order_depth.sell_orders[ask_price] == 0:
                del order_depth.sell_orders[ask_price]

    elif order.quantity < 0:
        remaining = -order.quantity
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if remaining <= 0:
                break
            if bid_price < order.price:
                break

            available = order_depth.buy_orders[bid_price]
            fill_qty = min(remaining, available)
            if fill_qty <= 0:
                continue

            fills.append(
                Trade(
                    symbol=product,
                    price=bid_price,
                    quantity=fill_qty,
                    buyer="BOOK",
                    seller="SUBMISSION",
                    timestamp=timestamp,
                )
            )

            remaining -= fill_qty
            pos -= fill_qty
            cash += bid_price * fill_qty
            order_depth.buy_orders[bid_price] -= fill_qty

            if order_depth.buy_orders[bid_price] == 0:
                del order_depth.buy_orders[bid_price]

    return fills, pos, cash


# =========================
# STATE BUILDING
# =========================
def build_state_for_timestamp(
    ts_prices: pd.DataFrame,
    ts_market_trades: pd.DataFrame,
    timestamp: int,
    trader_data: str,
    positions: Dict[str, int],
    own_trades_hist: Dict[str, List[Trade]],
    market_trades_hist: Dict[str, List[Trade]],
    simulate_extra_access: bool = False,
) -> TradingState:
    listings: Dict[str, Listing] = {}
    order_depths: Dict[str, OrderDepth] = {}

    for _, row in ts_prices.iterrows():
        product = row["product"]
        symbol = product

        listings[symbol] = Listing(
            symbol=symbol,
            product=product,
            denomination="XIREC",
        )

        od = build_order_depth(row)
        if simulate_extra_access:
            od = augment_order_depth_for_extra_access(od)

        order_depths[symbol] = od

    mt: Dict[str, List[Trade]] = {p: [] for p in ROUND_2_PRODUCTS}

    if not ts_market_trades.empty:
        for _, row in ts_market_trades.iterrows():
            symbol = row["symbol"]
            if symbol not in mt:
                mt[symbol] = []

            trade = Trade(
                symbol=symbol,
                price=int(row["price"]),
                quantity=int(row["quantity"]),
                buyer=(
                    str(row["buyer"])
                    if "buyer" in row and pd.notna(row["buyer"])
                    else None
                ),
                seller=(
                    str(row["seller"])
                    if "seller" in row and pd.notna(row["seller"])
                    else None
                ),
                timestamp=timestamp,
            )
            mt[symbol].append(trade)
            market_trades_hist.setdefault(symbol, []).append(trade)

    own_trades = {
        p: list(own_trades_hist.get(p, [])) for p in ROUND_2_PRODUCTS
    }
    market_trades = {
        p: list(market_trades_hist.get(p, [])) for p in ROUND_2_PRODUCTS
    }

    observations = Observation(
        plainValueObservations={},
        conversionObservations={},
    )

    return TradingState(
        traderData=trader_data,
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=dict(positions),
        observations=observations,
    )


# =========================
# PNL
# =========================
def mark_to_market(
    positions: Dict[str, int],
    cash: float,
    current_mid: Dict[str, Optional[float]],
) -> float:
    value = cash
    for product, pos in positions.items():
        mid = current_mid.get(product)
        if mid is not None:
            value += pos * mid
    return value


# =========================
# LOG BUILDERS
# =========================
def build_activities_log(
    prices_df: pd.DataFrame, equity_df: pd.DataFrame
) -> str:
    """
    Build a CSV-style activities log string similar to the host simulator.
    """
    merged = prices_df.merge(
        equity_df[["day", "timestamp", "equity"]],
        on=["day", "timestamp"],
        how="left",
    ).rename(columns={"equity": "profit_and_loss"})

    cols = [
        "day",
        "timestamp",
        "product",
        "bid_price_1",
        "bid_volume_1",
        "bid_price_2",
        "bid_volume_2",
        "bid_price_3",
        "bid_volume_3",
        "ask_price_1",
        "ask_volume_1",
        "ask_price_2",
        "ask_volume_2",
        "ask_price_3",
        "ask_volume_3",
        "mid_price",
        "profit_and_loss",
    ]

    for c in cols:
        if c not in merged.columns:
            merged[c] = ""

    merged = merged[cols].copy()
    merged = merged.fillna("")

    return merged.to_csv(sep=";", index=False, lineterminator="\n")


def build_trade_history(fills_df: pd.DataFrame) -> List[Dict]:
    """
    Convert local fills dataframe into simulator-like trade history entries.
    """
    if fills_df.empty:
        return []

    trade_history: List[Dict] = []

    for _, row in fills_df.iterrows():
        side = row["side"]
        trade_history.append(
            {
                "timestamp": int(row["timestamp"]),
                "buyer": "SUBMISSION" if side == "BUY" else "",
                "seller": "SUBMISSION" if side == "SELL" else "",
                "symbol": row["symbol"],
                "currency": "XIRECS",
                "price": float(row["price"]),
                "quantity": int(row["quantity"]),
            }
        )

    return trade_history


def write_simulator_style_log(
    output_path: str,
    prices_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    submission_id: Optional[str] = None,
) -> None:
    if submission_id is None:
        submission_id = str(uuid.uuid4())

    payload = {
        "submissionId": submission_id,
        "activitiesLog": build_activities_log(prices_df, equity_df),
        "tradeHistory": build_trade_history(fills_df),
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Trade log written to {output_path}")


# =========================
# BACKTEST
# =========================
def backtest(
    strategy_path: str,
    data_dir: str,
    assume_maf_accepted: bool = False,
    simulate_extra_access: bool = False,
    days: Optional[List[int]] = None,
    max_timestamp: Optional[int] = None,
):
    prices_df, trades_df = load_round2_data(
        data_dir=data_dir,
        days=days,
        max_timestamp=max_timestamp,
    )
    TraderClass = load_trader_class(strategy_path)
    trader = TraderClass()

    positions = {p: 0 for p in ROUND_2_PRODUCTS}
    cash = 0.0
    trader_data = ""

    own_trades_hist: Dict[str, List[Trade]] = {p: [] for p in ROUND_2_PRODUCTS}
    market_trades_hist: Dict[str, List[Trade]] = {
        p: [] for p in ROUND_2_PRODUCTS
    }

    last_mid_by_product: Dict[str, Optional[float]] = {
        p: None for p in ROUND_2_PRODUCTS
    }

    equity_curve = []
    fill_records = []

    grouped = prices_df.groupby(["day", "timestamp"], sort=True)

    for (day, timestamp), ts_prices in grouped:
        if not trades_df.empty:
            mask = (trades_df["day"] == day) & (
                trades_df["timestamp"] == timestamp
            )
            ts_market_trades = trades_df.loc[mask]
        else:
            ts_market_trades = pd.DataFrame()

        state = build_state_for_timestamp(
            ts_prices=ts_prices,
            ts_market_trades=ts_market_trades,
            timestamp=timestamp,
            trader_data=trader_data,
            positions=positions,
            own_trades_hist=own_trades_hist,
            market_trades_hist=market_trades_hist,
            simulate_extra_access=simulate_extra_access,
        )

        out = trader.run(state)

        if not isinstance(out, tuple) or len(out) != 3:
            raise ValueError(
                "Trader.run(state) must return (result, conversions, traderData)"
            )

        result, conversions, next_trader_data = out
        trader_data = next_trader_data

        for product, orders in result.items():
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]

            for order in orders:
                if order.symbol != product:
                    raise ValueError(
                        f"Order symbol mismatch: dict key={product}, order.symbol={order.symbol}"
                    )

                clipped_qty = clip_order_to_limits(
                    order=order,
                    product=product,
                    current_pos=positions[product],
                    limits=POSITION_LIMITS,
                )

                if clipped_qty == 0:
                    continue

                clipped_order = Order(
                    product, int(order.price), int(clipped_qty)
                )

                fills, new_pos, new_cash = execute_order_against_book(
                    order=clipped_order,
                    product=product,
                    order_depth=order_depth,
                    current_pos=positions[product],
                    cash=cash,
                    timestamp=timestamp,
                )

                if fills:
                    for f in fills:
                        own_trades_hist[product].append(f)
                        fill_records.append(
                            {
                                "day": day,
                                "timestamp": timestamp,
                                "symbol": product,
                                "price": f.price,
                                "quantity": f.quantity,
                                "side": (
                                    "BUY"
                                    if f.buyer == "SUBMISSION"
                                    else "SELL"
                                ),
                            }
                        )

                    positions[product] = new_pos
                    cash = new_cash

        current_mid = {}
        for _, row in ts_prices.iterrows():
            product = row["product"]

            od = build_order_depth(row)
            if simulate_extra_access:
                od = augment_order_depth_for_extra_access(od)

            mid = get_mark_price(row, od, last_mid_by_product[product])
            current_mid[product] = mid

            if mid is not None:
                last_mid_by_product[product] = mid

        equity_curve.append(
            {
                "day": day,
                "timestamp": timestamp,
                "cash": cash,
                "pos_osmium": positions["ASH_COATED_OSMIUM"],
                "pos_pepper": positions["INTARIAN_PEPPER_ROOT"],
                "equity": mark_to_market(positions, cash, current_mid),
            }
        )

    final_bid = trader.bid() if hasattr(trader, "bid") else 0
    raw_pnl = equity_curve[-1]["equity"] if equity_curve else 0.0
    net_pnl = raw_pnl - final_bid if assume_maf_accepted else raw_pnl

    equity_df = pd.DataFrame(equity_curve)
    fills_df = pd.DataFrame(fill_records)

    print("=" * 60)
    print("ROUND 2 BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Strategy file         : {strategy_path}")
    print(f"Data dir              : {data_dir}")
    print(
        f"Days used             : {days if days is not None else [-1, 0, 1]}"
    )
    print(f"Max timestamp         : {max_timestamp}")
    print(f"Trader bid()          : {final_bid}")
    print(f"MAF accepted?         : {assume_maf_accepted}")
    print(f"Simulate extra access : {simulate_extra_access}")
    print("-" * 60)
    print(f"Final cash            : {cash:.2f}")
    print("Final positions:")
    for p in ROUND_2_PRODUCTS:
        print(f"  {p}: {positions[p]}")
    print(f"Raw PnL               : {raw_pnl:.2f}")
    print(f"Net PnL               : {net_pnl:.2f}")
    print("=" * 60)

    if not fills_df.empty:
        print("\nFill summary by product:")
        print(
            fills_df.groupby("symbol")
            .agg(
                n_fills=("quantity", "count"),
                total_qty=("quantity", "sum"),
                avg_fill_px=("price", "mean"),
            )
            .to_string()
        )
    else:
        print("\nNo fills were generated.")

    return {
        "raw_pnl": raw_pnl,
        "net_pnl": net_pnl,
        "cash": cash,
        "positions": positions,
        "equity_curve": equity_df,
        "fills": fills_df,
        "prices_df": prices_df,
    }


# =========================
# PLOTTING
# =========================
def add_time_index(equity_df: pd.DataFrame) -> pd.DataFrame:
    df = equity_df.copy()
    unique_days = sorted(df["day"].unique())
    day_order = {day: i for i, day in enumerate(unique_days)}
    max_ts = df["timestamp"].max() + 1
    df["time_index"] = df["day"].map(day_order) * max_ts + df["timestamp"]
    return df


def plot_backtest_results(equity_df: pd.DataFrame):
    if equity_df.empty:
        print("No equity data to plot.")
        return

    df = add_time_index(equity_df)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["time_index"], df["equity"])
    axes[0].set_title("Total PnL / Equity Over Time")
    axes[0].set_ylabel("PnL")
    axes[0].grid(True)

    axes[1].plot(df["time_index"], df["pos_osmium"])
    axes[1].set_title("ASH_COATED_OSMIUM Position Over Time")
    axes[1].set_ylabel("Position")
    axes[1].grid(True)

    axes[2].plot(df["time_index"], df["pos_pepper"])
    axes[2].set_title("INTARIAN_PEPPER_ROOT Position Over Time")
    axes[2].set_ylabel("Position")
    axes[2].set_xlabel("Time")
    axes[2].grid(True)

    unique_days = sorted(df["day"].unique())
    max_ts = df["timestamp"].max() + 1
    for boundary_idx in range(1, len(unique_days)):
        boundary = boundary_idx * max_ts
        for ax in axes:
            ax.axvline(boundary, linestyle="--")

    plt.tight_layout()
    plt.show()


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", required=True, help="Path to strategy.py"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing round 2 csv files",
    )
    parser.add_argument(
        "--assume-maf-accepted",
        action="store_true",
        help="Subtract Trader.bid() from final PnL",
    )
    parser.add_argument(
        "--simulate-extra-access",
        action="store_true",
        help="Approximate +25% extra market access by augmenting visible book liquidity",
    )
    parser.add_argument(
        "--log-output",
        default="backtest_log.json",
        help="Path to write simulator-style trade log JSON",
    )
    parser.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=None,
        help="Days to include, e.g. --days 1 or --days -1 0 1",
    )
    parser.add_argument(
        "--max-timestamp",
        type=int,
        default=None,
        help="Maximum timestamp to include",
    )
    args = parser.parse_args()

    results = backtest(
        strategy_path=args.strategy,
        data_dir=args.data_dir,
        assume_maf_accepted=args.assume_maf_accepted,
        simulate_extra_access=args.simulate_extra_access,
        days=args.days,
        max_timestamp=args.max_timestamp,
    )

    plot_backtest_results(results["equity_curve"])

    write_simulator_style_log(
        output_path=args.log_output,
        prices_df=results["prices_df"],
        equity_df=results["equity_curve"],
        fills_df=results["fills"],
    )


if __name__ == "__main__":
    main()
