from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine import BacktestResult


def summarize_result(result: BacktestResult) -> pd.DataFrame:
    fills = pd.DataFrame([f.__dict__ for f in result.fills])
    if fills.empty:
        return pd.DataFrame([{
            'final_pnl': result.final_pnl,
            'num_fills': 0,
            'buy_fills': 0,
            'sell_fills': 0,
        }])

    return pd.DataFrame([{
        'final_pnl': result.final_pnl,
        'num_fills': len(fills),
        'buy_fills': int((fills['side'] == 'BUY').sum()),
        'sell_fills': int((fills['side'] == 'SELL').sum()),
        'aggressive_fills': int((fills['reason'] == 'aggressive').sum()),
        'passive_fills': int((fills['reason'] == 'passive').sum()),
    }])


def save_outputs(result: BacktestResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(out / 'equity_curve.csv', index=False)
    pd.DataFrame([f.__dict__ for f in result.fills]).to_csv(out / 'fills.csv', index=False)
    summarize_result(result).to_csv(out / 'summary.csv', index=False)
