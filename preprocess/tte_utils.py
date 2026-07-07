import csv
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def fmt_time(value: Optional[datetime]) -> str:
    return "" if value is None else value.strftime("%Y-%m-%d %H:%M:%S")


def time_to_days(start: datetime, end: datetime) -> float:
    return max((end - start).total_seconds() / 86400.0, 0.0)


def add_duration_fields(
    row: Dict[str, Any],
    prediction_time: datetime,
    event_time: Optional[datetime],
    censor_time: Optional[datetime],
    event_observed: bool,
    horizon_days: float,
) -> Optional[Dict[str, Any]]:
    horizon_time = prediction_time + timedelta(days=float(horizon_days))
    if event_time is not None and event_time <= prediction_time:
        return None
    if event_observed and event_time is not None and event_time <= horizon_time:
        observed_time = event_time
        observed = 1
    else:
        observed_time = min(
            [time for time in (censor_time, horizon_time) if time is not None]
        )
        observed = 0
    if observed_time <= prediction_time:
        return None
    row.update(
        {
            "prediction_time": fmt_time(prediction_time),
            "event_time": fmt_time(event_time if observed else None),
            "censor_time": fmt_time(observed_time if not observed else censor_time),
            "time_to_event": f"{time_to_days(prediction_time, observed_time):.6f}",
            "event_observed": observed,
            "horizon_days": f"{float(horizon_days):.6f}",
            "time_unit": "day",
        }
    )
    return row


def read_csv_records(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_records_by_key(records: Iterable[Any], key_fn) -> List[Tuple[str, List[Any]]]:
    grouped: Dict[str, List[Any]] = {}
    for record in records:
        key = str(key_fn(record))
        grouped.setdefault(key, []).append(record)
    return list(grouped.items())
