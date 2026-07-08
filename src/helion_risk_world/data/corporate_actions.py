"""Corporate action guards for BANKNIFTY constituent data — adapter over the
reusable quanthelion.dataquality.corporate_actions.flag_blackout_bars mechanism.

The HDFC merger date and blackout window are project-specific constants (kept
here, not in quanthelion); the generic "flag bars within N days of an event date"
mechanism is the reusable part, now implemented once in quanthelion. This adapter
preserves the original ``date_col``-based (non-index) call signature.

Original implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quanthelion.dataquality.corporate_actions import flag_blackout_bars

# HDFCBANK absorbed HDFC Ltd on 2023-07-01. BANKNIFTY rebalanced the same session.
HDFC_MERGER_DATE = date(2023, 7, 1)
BLACKOUT_DAYS = 5  # calendar days either side


def flag_merger_bars(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """Return boolean mask (True = unreliable) for bars near the HDFC merger."""
    dt_index = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
    tmp = pd.DataFrame(index=dt_index)
    flagged = flag_blackout_bars(tmp, [HDFC_MERGER_DATE], window_days=BLACKOUT_DAYS)
    return pd.Series(flagged["blackout_active"].to_numpy(), index=df.index)


def drop_merger_bars(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a copy of ``df`` with HDFC merger bars removed."""
    mask = flag_merger_bars(df, date_col)
    return df[~mask].copy()


__all__ = ["HDFC_MERGER_DATE", "BLACKOUT_DAYS", "flag_merger_bars", "drop_merger_bars"]
