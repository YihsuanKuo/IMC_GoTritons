# Round-0 Replay Backtester

Yes — breaking the code into several files is a good idea here. This project is split into:

- `backtester/datamodel.py`: minimal competition-style data classes
- `backtester/data_loader.py`: reads the historical CSV files and builds order-book snapshots
- `backtester/engine.py`: replay engine, matching logic, PnL accounting
- `backtester/reporting.py`: saves fills, equity curve, and a summary table
- `run_backtest.py`: command-line entry point
- `strategies/example_strategy.py`: a tiny sample strategy

## Backtesting assumptions

This backtester uses a **no-impact replay** model:

1. The historical market path is fixed.
2. Your orders affect **your fills, cash, and position**.
3. Your orders do **not** change the future market path.
4. Marketable orders fill immediately against visible top-of-book liquidity.
5. Passive orders rest for one snapshot interval and can fill conservatively from the trade tape during that interval.
6. Existing orders are treated as canceled/replaced on each new snapshot.

This is a reasonable first version for your competition data.

## Running it

From this folder:

```bash
python run_backtest.py \
  --prices /mnt/data/prices_round_0_day_-1.csv \
  --trades /mnt/data/trades_round_0_day_-1.csv \
  --strategy strategies/example_strategy.py \
  --output-dir output_day_minus_1
```

You can repeat for day `-2` by swapping the input files.

## Plugging in your own strategy

Your strategy file should expose a class named `Trader` with a method:

```python
class Trader:
    def run(self, state):
        return result, conversions, traderData
```

where `result` is a dictionary like:

```python
{
    'EMERALDS': [Order('EMERALDS', 9999, 5), Order('EMERALDS', 10001, -5)],
    'TOMATOES': [...],
}
```

## Output files

The backtester writes:

- `summary.csv`
- `fills.csv`
- `equity_curve.csv`

## Caveats

This is a **first-pass** replay engine, not a perfect simulator. The main simplifications are:

- passive fills are estimated from the trade tape without queue-position modeling
- no latency model
- no market impact model
- conversions are ignored for round 0

Those are all reasonable to leave out in v1.
