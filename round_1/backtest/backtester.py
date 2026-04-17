import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState


SUBMISSION = "SUBMISSION"
MARKET = "MARKET"


def _clean_number(value):
    if pd.isna(value):
        return None
    value = float(value)
    if value.is_integer():
        return int(value)
    return value


def load_strategy(strategy_path: Path):
    strategy_path = strategy_path.resolve()
    backtester_dir = Path(__file__).resolve().parent

    if str(backtester_dir) not in sys.path:
        sys.path.insert(0, str(backtester_dir))
    if str(strategy_path.parent) not in sys.path:
        sys.path.insert(0, str(strategy_path.parent))

    spec = importlib.util.spec_from_file_location("user_trader_module", strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {strategy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Trader"):
        raise AttributeError("Strategy file must define a Trader class.")

    return module.Trader()


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df['product'] = df['product'].astype(str).str.strip()
    return df.sort_values(['timestamp', 'product']).reset_index(drop=True)


def load_trade_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df['symbol'] = df['symbol'].astype(str).str.strip()
    return df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)


def build_order_depth(row: pd.Series) -> OrderDepth:
    buy_orders: Dict[float, int] = {}
    sell_orders: Dict[float, int] = {}

    for level in range(1, 4):
        bid_price = row.get(f'bid_price_{level}')
        bid_volume = row.get(f'bid_volume_{level}')
        if pd.notna(bid_price) and pd.notna(bid_volume) and int(bid_volume) != 0:
            buy_orders[_clean_number(bid_price)] = int(bid_volume)

        ask_price = row.get(f'ask_price_{level}')
        ask_volume = row.get(f'ask_volume_{level}')
        if pd.notna(ask_price) and pd.notna(ask_volume) and int(ask_volume) != 0:
            sell_orders[_clean_number(ask_price)] = -int(ask_volume)

    depth = OrderDepth()
    depth.buy_orders = buy_orders
    depth.sell_orders = sell_orders
    return depth


def build_market_trades(trades_df: pd.DataFrame) -> Dict[Tuple[int, str], List[Trade]]:
    grouped: Dict[Tuple[int, str], List[Trade]] = defaultdict(list)
    if trades_df.empty:
        return grouped

    for row in trades_df.itertuples(index=False):
        grouped[(int(row.timestamp), row.symbol)].append(
            Trade(
                symbol=row.symbol,
                price=_clean_number(row.price),
                quantity=int(row.quantity),
                buyer=None if pd.isna(row.buyer) else str(row.buyer),
                seller=None if pd.isna(row.seller) else str(row.seller),
                timestamp=int(row.timestamp),
            )
        )
    return grouped


def normalize_trader_output(output, previous_trader_data: str):
    if isinstance(output, tuple):
        if len(output) == 3:
            orders_by_product, conversions, trader_data = output
        elif len(output) == 2:
            orders_by_product, trader_data = output
            conversions = 0
        elif len(output) == 1:
            orders_by_product = output[0]
            conversions = 0
            trader_data = previous_trader_data
        else:
            raise ValueError('Unexpected Trader.run return signature.')
    else:
        orders_by_product = output
        conversions = 0
        trader_data = previous_trader_data

    if orders_by_product is None:
        orders_by_product = {}
    if trader_data is None:
        trader_data = ''

    return orders_by_product, conversions, str(trader_data)


def cap_order_quantity(order: Order, current_position: int, limit: Optional[int]) -> int:
    qty = int(order.quantity)
    if limit is None:
        return qty
    if qty > 0:
        max_buy = max(0, limit - current_position)
        return min(qty, max_buy)
    if qty < 0:
        max_sell = max(0, limit + current_position)
        return -min(-qty, max_sell)
    return 0


def derive_mid_price(row: pd.Series, depth: OrderDepth, previous_mid: Optional[float]) -> float:
    """Use the visible book when possible, then fall back safely.

    The round csv often stores mid_price=0 when one side of the book is missing.
    Marking inventory to zero creates artificial PnL cliffs, so we prefer:
    1) best-bid / best-ask midpoint when both sides are visible
    2) positive row mid_price if present
    3) previous valid mid for continuity
    4) single visible quote as a last resort
    5) zero only if nothing else exists
    """
    if depth.buy_orders and depth.sell_orders:
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    row_mid = row.get('mid_price')
    if pd.notna(row_mid) and float(row_mid) > 0:
        return float(row_mid)

    if previous_mid is not None and previous_mid > 0:
        return float(previous_mid)

    if depth.buy_orders:
        return float(max(depth.buy_orders.keys()))
    if depth.sell_orders:
        return float(min(depth.sell_orders.keys()))
    return 0.0


def match_orders_against_book(
    product: str,
    submitted_orders: Iterable[Order],
    depth: OrderDepth,
    timestamp: int,
    current_position: int,
    position_limit: Optional[int],
) -> Tuple[List[Trade], int, float, List[dict]]:
    fills: List[Trade] = []
    position = current_position
    cash_delta = 0.0
    fill_debug_rows: List[dict] = []

    buy_book = dict(depth.buy_orders)
    sell_book = dict(depth.sell_orders)

    for order in submitted_orders:
        qty = cap_order_quantity(order, position, position_limit)
        if qty == 0:
            continue

        limit_price = float(order.price)

        if qty > 0:
            remaining = qty
            for ask_price in sorted(list(sell_book.keys())):
                available = -sell_book[ask_price]
                if remaining <= 0:
                    break
                if ask_price > limit_price:
                    break
                if available <= 0:
                    continue

                fill_qty = min(remaining, available)
                trade = Trade(
                    symbol=product,
                    price=ask_price,
                    quantity=fill_qty,
                    buyer=SUBMISSION,
                    seller=MARKET,
                    timestamp=timestamp,
                )
                fills.append(trade)
                remaining -= fill_qty
                position += fill_qty
                cash_delta -= ask_price * fill_qty
                sell_book[ask_price] += fill_qty
                if sell_book[ask_price] == 0:
                    del sell_book[ask_price]
                fill_debug_rows.append({
                    'price': ask_price,
                    'quantity': fill_qty,
                    'side': 'BUY',
                    'position_after_fill_local': position,
                    'cash_delta_after_fill_local': cash_delta,
                })
        else:
            remaining = -qty
            for bid_price in sorted(list(buy_book.keys()), reverse=True):
                available = buy_book[bid_price]
                if remaining <= 0:
                    break
                if bid_price < limit_price:
                    break
                if available <= 0:
                    continue

                fill_qty = min(remaining, available)
                trade = Trade(
                    symbol=product,
                    price=bid_price,
                    quantity=fill_qty,
                    buyer=MARKET,
                    seller=SUBMISSION,
                    timestamp=timestamp,
                )
                fills.append(trade)
                remaining -= fill_qty
                position -= fill_qty
                cash_delta += bid_price * fill_qty
                buy_book[bid_price] -= fill_qty
                if buy_book[bid_price] == 0:
                    del buy_book[bid_price]
                fill_debug_rows.append({
                    'price': bid_price,
                    'quantity': fill_qty,
                    'side': 'SELL',
                    'position_after_fill_local': position,
                    'cash_delta_after_fill_local': cash_delta,
                })

    return fills, position, cash_delta, fill_debug_rows


class Backtester:
    def __init__(self, strategy, reset_each_day: bool = True):
        self.strategy = strategy
        self.reset_each_day = reset_each_day
        self.position_limits = getattr(strategy, 'POSITION_LIMITS', {})

    def run_day(
        self,
        price_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        day: int,
        starting_positions: Optional[Dict[str, int]] = None,
        starting_cash: float = 0.0,
        starting_trader_data: str = '',
    ):
        price_df = price_df.copy()
        trades_df = trades_df.copy()

        products = sorted(price_df['product'].unique())
        listings = {
            product: Listing(symbol=product, product=product, denomination='XIRECS')
            for product in products
        }
        market_trade_map = build_market_trades(trades_df)

        positions = defaultdict(int)
        if starting_positions:
            positions.update(starting_positions)

        cash = float(starting_cash)
        trader_data = starting_trader_data
        latest_mid: Dict[str, float] = {product: 0.0 for product in products}
        previous_own_trades: Dict[str, List[Trade]] = {product: [] for product in products}
        product_cash: Dict[str, float] = {product: 0.0 for product in products}

        pnl_rows = []
        fill_rows = []

        for timestamp in sorted(price_df['timestamp'].unique()):
            rows_at_timestamp = price_df.loc[price_df['timestamp'] == timestamp]

            order_depths: Dict[str, OrderDepth] = {}
            market_trades: Dict[str, List[Trade]] = {}

            for row_dict in rows_at_timestamp.to_dict(orient='records'):
                row = pd.Series(row_dict)
                product = row['product']
                depth = build_order_depth(row)
                order_depths[product] = depth
                latest_mid[product] = derive_mid_price(row, depth, latest_mid.get(product))
                market_trades[product] = list(market_trade_map.get((int(timestamp), product), []))

            state = TradingState(
                traderData=trader_data,
                timestamp=int(timestamp),
                listings=listings,
                order_depths=order_depths,
                own_trades={product: list(previous_own_trades.get(product, [])) for product in products},
                market_trades={product: list(market_trades.get(product, [])) for product in products},
                position=dict(positions),
                observations=Observation({}, {}),
            )

            raw_output = self.strategy.run(state)
            orders_by_product, conversions, trader_data = normalize_trader_output(raw_output, trader_data)
            current_own_trades: Dict[str, List[Trade]] = {product: [] for product in products}

            for product, orders in orders_by_product.items():
                if product not in order_depths or orders is None:
                    continue

                start_cash = cash
                start_position = positions[product]
                fills, new_position, cash_delta, fill_debug_rows = match_orders_against_book(
                    product=product,
                    submitted_orders=orders,
                    depth=order_depths[product],
                    timestamp=int(timestamp),
                    current_position=positions[product],
                    position_limit=self.position_limits.get(product),
                )

                positions[product] = new_position
                cash += cash_delta
                product_cash[product] += cash_delta
                current_own_trades[product] = fills

                local_cash = start_cash
                local_position = start_position
                for info in fill_debug_rows:
                    if info['side'] == 'BUY':
                        local_cash -= info['price'] * info['quantity']
                        local_position += info['quantity']
                    else:
                        local_cash += info['price'] * info['quantity']
                        local_position -= info['quantity']
                    fill_rows.append({
                        'day': day,
                        'timestamp': int(timestamp),
                        'product': product,
                        'price': info['price'],
                        'quantity': info['quantity'],
                        'side': info['side'],
                        'cash_after_fill': local_cash,
                        'position_after_fill': local_position,
                    })

            inventory_value = sum(positions[p] * latest_mid[p] for p in products)
            total_pnl = cash + inventory_value
            row = {
                'day': day,
                'timestamp': int(timestamp),
                'cash': cash,
                'inventory_value': inventory_value,
                'total_pnl': total_pnl,
                'conversions': conversions,
            }
            for product in products:
                row[f'position_{product}'] = positions[product]
                row[f'mid_{product}'] = latest_mid[product]
                row[f'mtm_{product}'] = positions[product] * latest_mid[product]
                row[f'cash_{product}'] = product_cash[product]
                row[f'pnl_{product}'] = product_cash[product] + positions[product] * latest_mid[product]
            pnl_rows.append(row)
            previous_own_trades = current_own_trades

        pnl_df = pd.DataFrame(pnl_rows)
        fills_df = pd.DataFrame(fill_rows)
        final_state = {
            'cash': cash,
            'positions': dict(positions),
            'trader_data': trader_data,
        }
        return pnl_df, fills_df, final_state

    def run_many_days(self, data_dir: Path, round_number: int, days: List[int]):
        all_pnl = []
        all_fills = []

        carry_positions: Dict[str, int] = {}
        carry_cash = 0.0
        carry_trader_data = ''

        for day in days:
            price_path = data_dir / f'prices_round_{round_number}_day_{day}.csv'
            trade_path = data_dir / f'trades_round_{round_number}_day_{day}.csv'
            if not price_path.exists():
                raise FileNotFoundError(f'Missing price file: {price_path}')
            if not trade_path.exists():
                raise FileNotFoundError(f'Missing trade file: {trade_path}')

            price_df = load_price_data(price_path)
            trades_df = load_trade_data(trade_path)

            if self.reset_each_day:
                start_positions = {}
                start_cash = 0.0
                start_trader_data = ''
            else:
                start_positions = carry_positions
                start_cash = carry_cash
                start_trader_data = carry_trader_data

            pnl_df, fills_df, final_state = self.run_day(
                price_df=price_df,
                trades_df=trades_df,
                day=day,
                starting_positions=start_positions,
                starting_cash=start_cash,
                starting_trader_data=start_trader_data,
            )
            all_pnl.append(pnl_df)
            all_fills.append(fills_df)
            carry_positions = final_state['positions']
            carry_cash = final_state['cash']
            carry_trader_data = final_state['trader_data']

        pnl_out = pd.concat(all_pnl, ignore_index=True) if all_pnl else pd.DataFrame()
        fills_out = pd.concat(all_fills, ignore_index=True) if all_fills else pd.DataFrame()
        return pnl_out, fills_out


def make_summary(pnl_df: pd.DataFrame, fills_df: pd.DataFrame) -> pd.DataFrame:
    if pnl_df.empty:
        return pd.DataFrame()
    summary_rows = []
    for day, grp in pnl_df.groupby('day', sort=True):
        final_row = grp.iloc[-1]
        summary_rows.append({
            'day': day,
            'final_cash': final_row['cash'],
            'final_inventory_value': final_row['inventory_value'],
            'final_total_pnl': final_row['total_pnl'],
            'num_timestamps': len(grp),
            'num_fills': 0 if fills_df.empty else int((fills_df['day'] == day).sum()),
        })

    final_row = pnl_df.sort_values(['day', 'timestamp']).iloc[-1]
    summary_rows.append({
        'day': 'ALL',
        'final_cash': final_row['cash'],
        'final_inventory_value': final_row['inventory_value'],
        'final_total_pnl': final_row['total_pnl'],
        'num_timestamps': int(len(pnl_df)),
        'num_fills': 0 if fills_df.empty else int(len(fills_df)),
    })
    return pd.DataFrame(summary_rows)


def parse_args():
    parser = argparse.ArgumentParser(description='Replay round data against a Trader strategy.')
    parser.add_argument('--strategy', type=Path, required=True, help='Path to the strategy file, e.g. pepper_root.py')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing prices_round_* and trades_round_* CSV files.')
    parser.add_argument('--round-number', type=int, default=1, help='Round number used in the CSV file names.')
    parser.add_argument('--days', type=int, nargs='+', default=[-2, -1, 0], help='Trading days to replay.')
    parser.add_argument('--output-dir', type=Path, default=Path('backtest_results'), help='Directory where results CSVs will be written.')
    parser.add_argument('--carry-state-across-days', action='store_true', help='If set, positions, cash, and traderData are carried from one day to the next.')
    return parser.parse_args()


def main():
    args = parse_args()
    strategy = load_strategy(args.strategy)
    backtester = Backtester(strategy=strategy, reset_each_day=not args.carry_state_across_days)
    pnl_df, fills_df = backtester.run_many_days(data_dir=args.data_dir, round_number=args.round_number, days=args.days)
    summary_df = make_summary(pnl_df, fills_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pnl_path = args.output_dir / 'pnl_timeseries.csv'
    fills_path = args.output_dir / 'fills.csv'
    summary_path = args.output_dir / 'summary.csv'
    pnl_df.to_csv(pnl_path, index=False)
    fills_df.to_csv(fills_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print('Backtest finished.')
    print(f'PnL time series: {pnl_path}')
    print(f'Fills: {fills_path}')
    print(f'Summary: {summary_path}')
    if not summary_df.empty:
        print() 
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
