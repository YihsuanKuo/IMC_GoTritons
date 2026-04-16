# Round 1 Backtester

This is a simple tutorial-style backtester for your `round_1` folder.

## Files

- `datamodel.py`: minimal classes used by the strategy and backtester.
- `backtester.py`: command-line backtester.
- `run_round1_backtest.py`: quick runner for the default `round_1` layout.

## Expected folder layout

```text
round_1/
├── pepper_root.py
├── datamodel.py
├── backtester.py
├── run_round1_backtest.py
└── data/
    ├── prices_round_1_day_-2.csv
    ├── prices_round_1_day_-1.csv
    ├── prices_round_1_day_0.csv
    ├── trades_round_1_day_-2.csv
    ├── trades_round_1_day_-1.csv
    └── trades_round_1_day_0.csv
```

## Quick start

Put these files into your `round_1` folder, then run:

```bash
python run_round1_backtest.py
```

That writes:

- `backtest_results/pnl_timeseries.csv`
- `backtest_results/fills.csv`
- `backtest_results/summary.csv`

## Flexible command-line usage

```bash
python backtester.py \
  --strategy pepper_root.py \
  --data-dir data \
  --round-number 1 \
  --days -2 -1 0 \
  --output-dir backtest_results
```

If you want to carry positions/cash/traderData across days instead of resetting flat each day:

```bash
python backtester.py \
  --strategy pepper_root.py \
  --data-dir data \
  --round-number 1 \
  --days -2 -1 0 \
  --output-dir backtest_results \
  --carry-state-across-days
```

## Important behavior

This backtester is intentionally simple:

- It only matches your orders against the visible order book in the current price snapshot.
- Unfilled quantities do **not** rest on the book.
- Buy orders execute against asks at or below your limit price.
- Sell orders execute against bids at or above your limit price.
- Position limits are enforced using `Trader.POSITION_LIMITS` if present.
- `state.market_trades` is populated from the round trade CSV at the current timestamp.
- `state.own_trades` contains fills from the previous timestamp.

So this is great for tutorial-style replay and quick strategy debugging, but it is not a full exchange simulator.
