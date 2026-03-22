#!/usr/bin/env python
"""
Pre/Post Trade Metric Study

CLI program to analyze backtest trades against ORATS historical core metrics.

Design goals for this small program:
- Keep architecture clean and right-sized
- Use ORATS client wrappers (no raw HTTP)
- Minimize API calls (1 get_hist_cores call per sampled ticker)
- Provide readable terminal output and optional CSV export
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_COLUMNS = ("date", "ticker", "exitDate", "profit", "leg")
CORE_FIELDS = "tradeDate,iv30d,orHv20d,contango,slope,slopepctile,ivPctile1y"

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 42
DEFAULT_TREND_THRESHOLD = 0.05


class StudyError(Exception):
    """User-friendly error for validation and runtime issues in this study."""


@dataclass
class AggregatedTrade:
    """Single trade after collapsing legs from the input CSV."""

    ticker: str
    entry_date: date
    exit_date: date
    pnl: float
    leg_count: int


@dataclass
class CoreSnapshot:
    """Single ORATS historical core row for a ticker/date."""

    trade_date: date
    iv30d: Optional[float]
    or_hv20d: Optional[float]
    contango: Optional[float]
    slope: Optional[float]
    slope_pctile_1y: Optional[float]
    iv_pctile_1y: Optional[float]
    iv_rv_spread: Optional[float]


@dataclass
class TickerSeries:
    """All ORATS snapshots for one ticker, sorted by trade_date."""

    ticker: str
    snapshots: List[CoreSnapshot]
    dates: List[date]


@dataclass
class TradeMetrics:
    """Computed metrics and flags for one sampled trade."""

    ticker: str
    pnl: float
    leg_count: int
    entry_date_input: date
    entry_date_used: Optional[date]
    exit_date_input: date
    exit_date_used: Optional[date]
    post_target_date_used: Optional[date]
    holding_trading_days: Optional[int]
    pre_days_available: int
    pre_days_with_spread: int
    insufficient_pre_entry: bool
    early_exit_used: bool
    post_window_truncated: bool
    iv_rv_spread_at_entry: Optional[float]
    iv_rv_spread_trend_coeff: Optional[float]
    spread_trend: Optional[str]
    contango_at_entry: Optional[float]
    slope_pctile_at_entry: Optional[float]
    iv_pctile1y_at_entry: Optional[float]
    spread_change_post_entry: Optional[float]
    contango_change_post_entry: Optional[float]
    segment: str = "Unassigned"
    notes: str = ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sampled backtest trades against ORATS historical core metrics.",
    )
    parser.add_argument(
        "--backtest",
        type=Path,
        help="Path to a backtest CSV. If omitted, an interactive picker is shown.",
    )
    parser.add_argument(
        "--backtests-dir",
        type=Path,
        default=Path("backtests"),
        help="Directory scanned for CSVs when --backtest is omitted (default: backtests).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of trades to sample (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used when sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Ignore --seed and use non-deterministic random sampling.",
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=DEFAULT_TREND_THRESHOLD,
        help="Threshold for classifying spread trend (default: 0.05).",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Optional path to export per-trade computed metrics as CSV.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional run diagnostics.",
    )
    return parser.parse_args(argv)


def _normalize_columns(fieldnames: Optional[Iterable[str]]) -> Dict[str, str]:
    if not fieldnames:
        raise StudyError("Input CSV appears to be missing a header row.")

    normalized: Dict[str, str] = {}
    for raw_name in fieldnames:
        key = raw_name.strip().lower()
        if key:
            normalized[key] = raw_name

    missing = [c for c in REQUIRED_COLUMNS if c.lower() not in normalized]
    if missing:
        raise StudyError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
            + f". Required columns are: {', '.join(REQUIRED_COLUMNS)}"
        )
    return normalized


def _parse_iso_date(raw: str, column: str, row_num: int) -> date:
    text = (raw or "").strip()
    if not text:
        raise StudyError(f"Row {row_num}: missing value for '{column}'.")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise StudyError(
            f"Row {row_num}: invalid {column} '{text}'. Expected YYYY-MM-DD."
        ) from exc


def _parse_profit(raw: str, row_num: int) -> float:
    text = (raw or "").strip()
    if not text:
        raise StudyError(f"Row {row_num}: missing value for 'profit'.")

    neg = False
    if text.startswith("(") and text.endswith(")"):
        neg = True
        text = text[1:-1].strip()

    text = text.replace("$", "").replace(",", "")
    if text.endswith("%"):
        text = text[:-1]

    try:
        value = float(text)
    except ValueError as exc:
        raise StudyError(f"Row {row_num}: invalid numeric profit '{raw}'.") from exc

    return -value if neg else value


def choose_backtest_file(backtest: Optional[Path], backtests_dir: Path) -> Path:
    if backtest is not None:
        if not backtest.exists() or not backtest.is_file():
            raise StudyError(f"Backtest file not found: {backtest}")
        return backtest

    csv_files = sorted(p for p in backtests_dir.glob("*.csv") if p.is_file())
    if not csv_files:
        raise StudyError(
            f"No CSV files found in '{backtests_dir}'. "
            "Add a backtest file there or pass --backtest <path>."
        )

    if not sys.stdin.isatty():
        if len(csv_files) == 1:
            return csv_files[0]
        raise StudyError(
            "Multiple backtest files found but interactive selection is unavailable. "
            "Pass --backtest <path>."
        )

    print("\nAvailable backtest files:\n")
    for idx, file_path in enumerate(csv_files, start=1):
        print(f"  {idx:>2}. {file_path.name}")

    while True:
        choice = input("\nSelect a file number (or 'q' to quit): ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            raise StudyError("Selection cancelled by user.")
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        selected_index = int(choice)
        if 1 <= selected_index <= len(csv_files):
            return csv_files[selected_index - 1]
        print(f"Please enter a number between 1 and {len(csv_files)}.")


def load_and_aggregate_trades(csv_path: Path) -> List[AggregatedTrade]:
    grouped: Dict[Tuple[str, date], AggregatedTrade] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        columns = _normalize_columns(reader.fieldnames)

        date_col = columns["date"]
        ticker_col = columns["ticker"]
        exit_col = columns["exitdate"]
        profit_col = columns["profit"]
        leg_col = columns["leg"]

        for row_num, row in enumerate(reader, start=2):
            entry_date = _parse_iso_date(row.get(date_col, ""), date_col, row_num)
            exit_date = _parse_iso_date(row.get(exit_col, ""), exit_col, row_num)
            if exit_date < entry_date:
                raise StudyError(
                    f"Row {row_num}: exitDate ({exit_date}) is before date ({entry_date})."
                )

            ticker = (row.get(ticker_col, "") or "").strip().upper()
            if not ticker:
                raise StudyError(f"Row {row_num}: missing value for '{ticker_col}'.")

            leg_value = (row.get(leg_col, "") or "").strip()
            if not leg_value:
                raise StudyError(f"Row {row_num}: missing value for '{leg_col}'.")

            pnl = _parse_profit(row.get(profit_col, ""), row_num)
            key = (ticker, entry_date)

            if key not in grouped:
                grouped[key] = AggregatedTrade(
                    ticker=ticker,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    pnl=pnl,
                    leg_count=1,
                )
                continue

            aggregate = grouped[key]
            if aggregate.exit_date != exit_date:
                raise StudyError(
                    f"Row {row_num}: inconsistent exitDate for trade {ticker} {entry_date}. "
                    f"Expected {aggregate.exit_date}, got {exit_date}."
                )
            aggregate.pnl += pnl
            aggregate.leg_count += 1

    trades = sorted(grouped.values(), key=lambda t: (t.entry_date, t.ticker))
    if not trades:
        raise StudyError(f"No trade rows found in {csv_path}.")

    return trades


def sample_trades(
    trades: Sequence[AggregatedTrade],
    sample_size: int,
    seed: int,
    randomize: bool,
) -> Tuple[List[AggregatedTrade], str]:
    if sample_size <= 0:
        raise StudyError("--sample-size must be a positive integer.")
    if sample_size > len(trades):
        raise StudyError(
            f"Requested sample size {sample_size}, but only {len(trades)} aggregated trades are available."
        )

    if randomize:
        rng = random.Random()
        seed_label = "randomized (system entropy)"
    else:
        rng = random.Random(seed)
        seed_label = f"fixed ({seed})"

    selected = rng.sample(list(trades), sample_size)
    selected.sort(key=lambda t: (t.entry_date, t.ticker))
    return selected, seed_label


def _to_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_hist_core_series(
    tickers: Sequence[str],
    debug: bool,
) -> Tuple[Dict[str, TickerSeries], int]:
    """
    Fetch historical core rows with one API call per ticker.

    Uses get_hist_cores(..., fields=<comma-delimited string>) to keep payloads tight.
    """
    try:
        from orats_client import OratsClient, OratsError
        from orats_client.endpoints import get_hist_cores
    except ImportError as exc:
        raise StudyError(
            "Could not import 'orats_client'. Install local editable dependency first:\n"
            "  pip install -e ../orats-client"
        ) from exc

    try:
        client = OratsClient()
    except ValueError as exc:
        raise StudyError(
            "ORATS API key is not configured. Set ORATS_API_KEY in your environment."
        ) from exc

    series_by_ticker: Dict[str, TickerSeries] = {}
    call_count = 0

    with client:
        for ticker in tickers:
            call_count += 1
            try:
                rows = get_hist_cores(client, ticker=ticker, fields=CORE_FIELDS)
            except OratsError as exc:
                raise StudyError(
                    f"ORATS API error {exc.status_code} while fetching ticker '{ticker}': {exc.message}"
                ) from exc
            except ValueError as exc:
                raise StudyError(
                    f"Validation error while requesting ORATS data for '{ticker}': {exc}"
                ) from exc
            except Exception as exc:
                raise StudyError(
                    f"Failed to fetch ORATS data for '{ticker}': {exc}"
                ) from exc

            snapshots: List[CoreSnapshot] = []
            for row in rows:
                raw_trade_date = getattr(row, "tradeDate", None)
                if not raw_trade_date:
                    continue

                try:
                    trade_date = date.fromisoformat(str(raw_trade_date))
                except ValueError:
                    continue

                iv30d = _to_optional_float(getattr(row, "iv30d", None))
                or_hv20d = _to_optional_float(getattr(row, "orHv20d", None))
                contango = _to_optional_float(getattr(row, "contango", None))
                slope = _to_optional_float(getattr(row, "slope", None))
                slope_pctile_1y = _to_optional_float(getattr(row, "slopepctile", None))
                iv_pctile_1y = _to_optional_float(getattr(row, "ivPctile1y", None))

                iv_rv_spread: Optional[float] = None
                if iv30d is not None and or_hv20d is not None:
                    iv_rv_spread = iv30d - or_hv20d

                snapshots.append(
                    CoreSnapshot(
                        trade_date=trade_date,
                        iv30d=iv30d,
                        or_hv20d=or_hv20d,
                        contango=contango,
                        slope=slope,
                        slope_pctile_1y=slope_pctile_1y,
                        iv_pctile_1y=iv_pctile_1y,
                        iv_rv_spread=iv_rv_spread,
                    )
                )

            snapshots.sort(key=lambda s: s.trade_date)
            series_by_ticker[ticker] = TickerSeries(
                ticker=ticker,
                snapshots=snapshots,
                dates=[s.trade_date for s in snapshots],
            )

            if debug:
                print(
                    f"[debug] fetched {len(snapshots)} hist core rows for {ticker} "
                    f"(fields={CORE_FIELDS})"
                )

    return series_by_ticker, call_count


def nearest_index(dates: Sequence[date], target: date) -> Optional[int]:
    if not dates:
        return None

    idx = bisect_left(dates, target)
    if idx == 0:
        return 0
    if idx >= len(dates):
        return len(dates) - 1

    before = dates[idx - 1]
    after = dates[idx]
    if (target - before) <= (after - target):
        return idx - 1
    return idx


def linear_regression_slope(values: Sequence[float]) -> Optional[float]:
    """
    Simple OLS slope for y ~ x where x is 0..n-1.
    Returns None if fewer than 2 points.
    """
    n = len(values)
    if n < 2:
        return None

    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    numerator = 0.0
    denominator = 0.0

    for i, y_val in enumerate(values):
        x_delta = i - mean_x
        y_delta = y_val - mean_y
        numerator += x_delta * y_delta
        denominator += x_delta * x_delta

    if denominator == 0.0:
        return None
    return numerator / denominator


def classify_trend(slope_coeff: Optional[float], threshold: float) -> Optional[str]:
    if slope_coeff is None:
        return None
    if slope_coeff > threshold:
        return "Rising"
    if slope_coeff < -threshold:
        return "Falling"
    return "Flat"


def analyze_trade(
    trade: AggregatedTrade,
    ticker_series: Optional[TickerSeries],
    trend_threshold: float,
) -> TradeMetrics:
    notes: List[str] = []

    if ticker_series is None or not ticker_series.snapshots:
        notes.append("No ORATS historical core data available for ticker.")
        return TradeMetrics(
            ticker=trade.ticker,
            pnl=trade.pnl,
            leg_count=trade.leg_count,
            entry_date_input=trade.entry_date,
            entry_date_used=None,
            exit_date_input=trade.exit_date,
            exit_date_used=None,
            post_target_date_used=None,
            holding_trading_days=None,
            pre_days_available=0,
            pre_days_with_spread=0,
            insufficient_pre_entry=True,
            early_exit_used=False,
            post_window_truncated=False,
            iv_rv_spread_at_entry=None,
            iv_rv_spread_trend_coeff=None,
            spread_trend=None,
            contango_at_entry=None,
            slope_pctile_at_entry=None,
            iv_pctile1y_at_entry=None,
            spread_change_post_entry=None,
            contango_change_post_entry=None,
            notes="; ".join(notes),
        )

    entry_idx = nearest_index(ticker_series.dates, trade.entry_date)
    exit_idx = nearest_index(ticker_series.dates, trade.exit_date)

    if entry_idx is None or exit_idx is None:
        notes.append("Unable to align entry/exit date to ORATS trade dates.")
        return TradeMetrics(
            ticker=trade.ticker,
            pnl=trade.pnl,
            leg_count=trade.leg_count,
            entry_date_input=trade.entry_date,
            entry_date_used=None,
            exit_date_input=trade.exit_date,
            exit_date_used=None,
            post_target_date_used=None,
            holding_trading_days=None,
            pre_days_available=0,
            pre_days_with_spread=0,
            insufficient_pre_entry=True,
            early_exit_used=False,
            post_window_truncated=False,
            iv_rv_spread_at_entry=None,
            iv_rv_spread_trend_coeff=None,
            spread_trend=None,
            contango_at_entry=None,
            slope_pctile_at_entry=None,
            iv_pctile1y_at_entry=None,
            spread_change_post_entry=None,
            contango_change_post_entry=None,
            notes="; ".join(notes),
        )

    if exit_idx < entry_idx:
        # Defensive guard in case nearest-date alignment moves exit left of entry.
        exit_idx = entry_idx
        notes.append("Exit date aligned before entry; adjusted to entry date.")

    snapshots = ticker_series.snapshots
    entry_snapshot = snapshots[entry_idx]
    exit_snapshot = snapshots[exit_idx]

    pre_start = max(0, entry_idx - 10)
    pre_window = snapshots[pre_start:entry_idx]
    pre_days_available = len(pre_window)
    insufficient_pre_entry = pre_days_available < 10
    if insufficient_pre_entry:
        notes.append("Fewer than 10 pre-entry trading days available.")

    pre_spreads = [s.iv_rv_spread for s in pre_window if s.iv_rv_spread is not None]
    trend_coeff = linear_regression_slope(pre_spreads)
    if len(pre_spreads) < 2:
        notes.append("Insufficient valid pre-entry ivRvSpread points for regression.")
    spread_trend = classify_trend(trend_coeff, trend_threshold)

    post_window_truncated = entry_idx + 10 >= len(snapshots)
    ten_day_post_idx = min(entry_idx + 10, len(snapshots) - 1)

    if exit_idx <= ten_day_post_idx:
        target_idx = exit_idx
        early_exit_used = exit_idx < ten_day_post_idx
    else:
        target_idx = ten_day_post_idx
        early_exit_used = False

    if early_exit_used:
        notes.append("Used exit date because trade exited before +10 trading days.")
    if post_window_truncated:
        notes.append("+10 trading-day post window truncated by available ORATS history.")

    target_snapshot = snapshots[target_idx]

    iv_rv_spread_at_entry = entry_snapshot.iv_rv_spread
    if iv_rv_spread_at_entry is None:
        notes.append("Missing iv30d/orHv20d at entry; spread metrics may be null.")

    spread_change_post_entry: Optional[float] = None
    if (
        iv_rv_spread_at_entry is not None
        and target_snapshot.iv_rv_spread is not None
    ):
        spread_change_post_entry = target_snapshot.iv_rv_spread - iv_rv_spread_at_entry

    contango_at_entry = entry_snapshot.contango
    contango_change_post_entry: Optional[float] = None
    if contango_at_entry is not None and target_snapshot.contango is not None:
        contango_change_post_entry = target_snapshot.contango - contango_at_entry
    else:
        notes.append("Contango missing at entry or post target date.")

    slope_pctile_at_entry = entry_snapshot.slope_pctile_1y
    if slope_pctile_at_entry is None:
        notes.append("Missing slopepctile at entry.")

    iv_pctile1y_at_entry = entry_snapshot.iv_pctile_1y
    if iv_pctile1y_at_entry is None:
        notes.append("Missing ivPctile1y at entry.")

    return TradeMetrics(
        ticker=trade.ticker,
        pnl=trade.pnl,
        leg_count=trade.leg_count,
        entry_date_input=trade.entry_date,
        entry_date_used=entry_snapshot.trade_date,
        exit_date_input=trade.exit_date,
        exit_date_used=exit_snapshot.trade_date,
        post_target_date_used=target_snapshot.trade_date,
        holding_trading_days=target_idx - entry_idx,
        pre_days_available=pre_days_available,
        pre_days_with_spread=len(pre_spreads),
        insufficient_pre_entry=insufficient_pre_entry,
        early_exit_used=early_exit_used,
        post_window_truncated=post_window_truncated,
        iv_rv_spread_at_entry=iv_rv_spread_at_entry,
        iv_rv_spread_trend_coeff=trend_coeff,
        spread_trend=spread_trend,
        contango_at_entry=contango_at_entry,
        slope_pctile_at_entry=slope_pctile_at_entry,
        iv_pctile1y_at_entry=iv_pctile1y_at_entry,
        spread_change_post_entry=spread_change_post_entry,
        contango_change_post_entry=contango_change_post_entry,
        notes="; ".join(notes),
    )


def analyze_trades(
    sampled_trades: Sequence[AggregatedTrade],
    series_by_ticker: Dict[str, TickerSeries],
    trend_threshold: float,
) -> List[TradeMetrics]:
    analyzed: List[TradeMetrics] = []
    for trade in sampled_trades:
        analyzed.append(
            analyze_trade(
                trade=trade,
                ticker_series=series_by_ticker.get(trade.ticker),
                trend_threshold=trend_threshold,
            )
        )
    return analyzed


def segment_trades(
    metrics: Sequence[TradeMetrics],
) -> Tuple[List[TradeMetrics], List[TradeMetrics], List[TradeMetrics]]:
    ordered = sorted(metrics, key=lambda m: m.pnl, reverse=True)
    n = len(ordered)
    if n < 3:
        raise StudyError("Need at least 3 sampled trades to segment into groups.")

    if n == 100:
        winners_n = 33
        losers_n = 33
    else:
        winners_n = n // 3
        losers_n = n // 3

    if winners_n == 0 or losers_n == 0:
        raise StudyError(
            "Sample size is too small to form winners/losers groups. "
            "Increase --sample-size."
        )

    winners = ordered[:winners_n]
    losers = ordered[-losers_n:]
    middle = ordered[winners_n : n - losers_n]

    for row in winners:
        row.segment = "Winner"
    for row in losers:
        row.segment = "Loser"
    for row in middle:
        row.segment = "Middle"

    return winners, middle, losers


def mean_or_none(values: Sequence[Optional[float]]) -> Tuple[Optional[float], int]:
    usable = [v for v in values if v is not None]
    if not usable:
        return None, 0
    return sum(usable) / len(usable), len(usable)


def format_number(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    str_rows: List[List[str]] = [[str(cell) for cell in row] for row in rows]
    widths: List[int] = []
    for col_idx, header in enumerate(headers):
        max_cell = max((len(row[col_idx]) for row in str_rows), default=0)
        widths.append(max(len(header), max_cell))

    def _format_row(cells: Sequence[str]) -> str:
        padded = [cells[i].ljust(widths[i]) for i in range(len(widths))]
        return "| " + " | ".join(padded) + " |"

    line = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    output_lines = [line, _format_row([str(h) for h in headers]), line]
    for row in str_rows:
        output_lines.append(_format_row(row))
    output_lines.append(line)
    return "\n".join(output_lines)


def build_summary_comparison_rows(
    winners: Sequence[TradeMetrics],
    losers: Sequence[TradeMetrics],
) -> List[List[str]]:
    metric_specs = [
        ("IV-RV Spread at Entry", "iv_rv_spread_at_entry"),
        ("IV-RV Spread Trend Coefficient", "iv_rv_spread_trend_coeff"),
        ("Contango at Entry", "contango_at_entry"),
        ("Slope Percentile at Entry", "slope_pctile_at_entry"),
        ("IV Percentile at Entry", "iv_pctile1y_at_entry"),
        ("IV-RV Spread Change (10d post)", "spread_change_post_entry"),
        ("Contango Change (10d post)", "contango_change_post_entry"),
    ]

    rows: List[List[str]] = []
    for label, field_name in metric_specs:
        winner_avg, _ = mean_or_none([getattr(m, field_name) for m in winners])
        loser_avg, _ = mean_or_none([getattr(m, field_name) for m in losers])
        diff = winner_avg - loser_avg if winner_avg is not None and loser_avg is not None else None
        rows.append(
            [
                label,
                format_number(winner_avg),
                format_number(loser_avg),
                format_number(diff),
            ]
        )
    return rows


def build_trend_distribution_rows(
    winners: Sequence[TradeMetrics],
    losers: Sequence[TradeMetrics],
) -> List[List[str]]:
    categories = ("Rising", "Flat", "Falling")
    rows: List[List[str]] = []
    for category in categories:
        winner_count = sum(1 for m in winners if m.spread_trend == category)
        loser_count = sum(1 for m in losers if m.spread_trend == category)
        rows.append([category, str(winner_count), str(loser_count)])
    return rows


def print_report(
    *,
    backtest_file: Path,
    total_trades: int,
    sampled: Sequence[TradeMetrics],
    seed_label: str,
    unique_tickers: Sequence[str],
    api_call_count: int,
    winners: Sequence[TradeMetrics],
    middle: Sequence[TradeMetrics],
    losers: Sequence[TradeMetrics],
) -> None:
    summary_rows = build_summary_comparison_rows(winners, losers)
    trend_rows = build_trend_distribution_rows(winners, losers)

    insufficient_pre = sum(1 for m in sampled if m.insufficient_pre_entry)
    early_exit = sum(1 for m in sampled if m.early_exit_used)
    no_orats_data = sum(1 for m in sampled if m.entry_date_used is None)
    missing_entry_spread = sum(1 for m in sampled if m.iv_rv_spread_at_entry is None)
    missing_post_spread = sum(1 for m in sampled if m.spread_change_post_entry is None)
    missing_post_contango = sum(1 for m in sampled if m.contango_change_post_entry is None)

    print("\n=== Pre/Post Trade Metric Study ===")
    print(f"Backtest file             : {backtest_file}")
    print(f"Aggregated trades loaded  : {total_trades}")
    print(f"Sampled trades analyzed   : {len(sampled)}")
    print(f"Sampling mode             : {seed_label}")
    print(f"Unique sampled tickers    : {len(unique_tickers)} ({', '.join(unique_tickers)})")
    print(f"ORATS API calls           : {api_call_count} (1 get_hist_cores call per ticker)")
    print(f"Segments                  : winners={len(winners)} middle={len(middle)} losers={len(losers)}")

    print("\nEdge-case / data-quality summary:")
    print(f"- Insufficient pre-entry history (<10 days): {insufficient_pre}")
    print(f"- Early exit used for post-entry calculations: {early_exit}")
    print(f"- Trades with no ORATS rows for ticker: {no_orats_data}")
    print(f"- Missing IV-RV spread at entry: {missing_entry_spread}")
    print(f"- Missing IV-RV spread change metric: {missing_post_spread}")
    print(f"- Missing contango change metric: {missing_post_contango}")

    print("\nSummary Comparison Table")
    print(
        render_table(
            headers=["Metric", "Winners Avg", "Losers Avg", "Difference"],
            rows=summary_rows,
        )
    )

    print("\nTrend Distribution Table")
    print(
        render_table(
            headers=["Spread Trend", "Winners Count", "Losers Count"],
            rows=trend_rows,
        )
    )


def _serialize_for_csv(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def export_metrics_csv(metrics: Sequence[TradeMetrics], export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return

    fieldnames = list(asdict(metrics[0]).keys())
    with export_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            raw = asdict(metric)
            serialized = {k: _serialize_for_csv(v) for k, v in raw.items()}
            writer.writerow(serialized)


def run(args: argparse.Namespace) -> int:
    backtest_file = choose_backtest_file(args.backtest, args.backtests_dir)
    trades = load_and_aggregate_trades(backtest_file)
    sampled_trades, seed_label = sample_trades(
        trades=trades,
        sample_size=args.sample_size,
        seed=args.seed,
        randomize=args.randomize,
    )

    unique_tickers = sorted({t.ticker for t in sampled_trades})
    series_by_ticker, api_call_count = fetch_hist_core_series(unique_tickers, debug=args.debug)

    metrics = analyze_trades(
        sampled_trades=sampled_trades,
        series_by_ticker=series_by_ticker,
        trend_threshold=args.trend_threshold,
    )
    winners, middle, losers = segment_trades(metrics)

    print_report(
        backtest_file=backtest_file,
        total_trades=len(trades),
        sampled=metrics,
        seed_label=seed_label,
        unique_tickers=unique_tickers,
        api_call_count=api_call_count,
        winners=winners,
        middle=middle,
        losers=losers,
    )

    if args.export_csv:
        export_metrics_csv(metrics, args.export_csv)
        print(f"\nExported per-trade metrics to: {args.export_csv}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except StudyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - defensive fallback
        if args.debug:
            raise
        print(f"Unexpected error: {exc}", file=sys.stderr)
        print("Re-run with --debug for full traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
