import io
import re
import warnings as _warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import pandas as pd


class IngestError(Exception):
    """Exception raised when validation fails critically and processing cannot continue."""
    pass


@dataclass
class IngestResult:
    series: np.ndarray
    dates: Optional[List[str]] = None
    frequency: str = "unknown"
    seasonal_period: int = 1
    warnings: List[str] = field(default_factory=list)
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    available_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series": self.series.tolist(),
            "dates": self.dates,
            "frequency": self.frequency,
            "seasonal_period": self.seasonal_period,
            "warnings": self.warnings,
            "date_column": self.date_column,
            "value_column": self.value_column,
            "available_columns": self.available_columns,
            "n": len(self.series),
        }


# --------------------------------------------------------------------------
# Format sniffing helpers
#
# The guiding rule: a bare number is NEVER a date. pandas will happily read
# 5200 as an epoch timestamp, which used to make single-column numeric input
# impossible to parse. Date detection is therefore pattern-driven, not
# exception-driven.
# --------------------------------------------------------------------------

_DATE_PATTERNS = [
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",          # 2023-01-01, 2023/01/01
    r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",       # 01/15/2023, 15-01-2023
    r"^\d{4}[-\s]?Q[1-4]$",                   # 2022-Q1, 2022Q1
    r"^Q[1-4][-\s]?\d{4}$",                   # Q1-2022
    r"^\d{4}[-/]\d{1,2}$",                    # 2023-01
    r"^[A-Za-z]{3,9}[-\s]\d{2,4}$",           # Jan-2024, January 2024
    r"^\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}$",  # 01-Jan-2024
    r"^[A-Za-z]{3,9}\s\d{1,2},?\s\d{4}$",     # January 5, 2024
]

# Everything that is not part of a number: currency symbols, thousands
# separators, percent signs, stray whitespace.
_NON_NUMERIC_CHARS = re.compile(r"[^\d.\-+eE]")
_PARENTHESIZED = re.compile(r"^\((.*)\)$")


def _to_datetime(values) -> pd.Series:
    """pd.to_datetime over user data, without the per-element format warning.

    Business exports legitimately mix date spellings; falling back to dateutil
    per element is the behaviour we want, so the warning is noise.
    """
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce")


def _is_date_like_token(token: str) -> bool:
    token = str(token).strip().strip('"').strip("'")
    if not token:
        return False
    return any(re.match(p, token, re.IGNORECASE) for p in _DATE_PATTERNS)


def _clean_scalar_numeric(token: str) -> Optional[float]:
    """Parse a single cell as a number, tolerating $ , % and (1,200) negatives."""
    token = str(token).strip().strip('"').strip("'")
    if not token:
        return None
    negative = bool(_PARENTHESIZED.match(token))
    if negative:
        token = _PARENTHESIZED.sub(r"\1", token)
    token = _NON_NUMERIC_CHARS.sub("", token)
    if token in ("", "-", "+", ".", "-.", "+."):
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value


def _clean_numeric_column(s: pd.Series) -> pd.Series:
    """Vectorised _clean_scalar_numeric over a column."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return s.map(lambda v: _clean_scalar_numeric(v) if pd.notna(v) else np.nan).astype(float)


def _is_date_like_column(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        # The only numeric column that counts as dates is a plain year column.
        values = pd.to_numeric(s, errors="coerce").dropna()
        if len(values) == 0:
            return False
        return bool((values % 1 == 0).all() and values.between(1900, 2100).all())
    sample = s.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(20)
    if len(sample) == 0:
        return False
    return sample.map(_is_date_like_token).mean() >= 0.7


def _row_is_header(cells: List[Any]) -> bool:
    """A row is a header only if no cell reads as a number or a date."""
    meaningful = [c for c in cells if str(c).strip() not in ("", "nan", "None")]
    if not meaningful:
        return False
    for cell in meaningful:
        if _clean_scalar_numeric(cell) is not None:
            return False
        if _is_date_like_token(cell):
            return False
    return True


def _sniff_separator(lines: List[str]) -> Optional[str]:
    """Pick the delimiter that splits rows into a consistent column count."""
    best: Optional[Tuple[str, int]] = None
    probe = lines[:50]
    for sep in [",", "\t", ";", "|"]:
        counts = [line.count(sep) for line in probe]
        if min(counts) < 1:
            continue
        mode = max(set(counts), key=counts.count)
        if mode >= 1 and counts.count(mode) >= 0.9 * len(probe):
            if best is None or mode > best[1]:
                best = (sep, mode)
    return best[0] if best else None


def _read_table(text: str) -> pd.DataFrame:
    """Parse pasted text or CSV bytes into a DataFrame without guessing wrong."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in normalized.split("\n") if line.strip()]
    if not lines:
        raise IngestError(
            "Nothing to read — the input is empty. Paste your numbers, or choose a file."
        )

    separator = _sniff_separator(lines)

    # Single column: every line is one value. No delimiter, no date column.
    if separator is None:
        header_present = _row_is_header([lines[0]])
        body = lines[1:] if header_present else lines
        name = lines[0].strip().strip('"') if header_present else "value"
        return pd.DataFrame({name or "value": [line.strip() for line in body]})

    first_cells = [c.strip().strip('"') for c in lines[0].split(separator)]
    header_present = _row_is_header(first_cells)

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=separator,
        header=0 if header_present else None,
        engine="python",
        skipinitialspace=True,
    )
    if not header_present:
        df.columns = [f"column_{i + 1}" for i in range(len(df.columns))]
    return df


def _read_excel(content: bytes) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content), header=None)
    if raw.empty:
        raise IngestError("That spreadsheet appears to be empty.")
    if _row_is_header(raw.iloc[0].tolist()):
        df = raw[1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in raw.iloc[0].tolist()]
    else:
        df = raw
        df.columns = [f"column_{i + 1}" for i in range(len(df.columns))]
    return df


def infer_frequency_and_seasonality(dates: pd.Series) -> Tuple[str, int, List[str]]:
    warnings = []
    if dates is None or len(dates) < 2:
        warnings.append("No dates found — seasonal patterns won't be detected.")
        return "unknown", 1, warnings

    # Ensure datetime
    dt_series = _to_datetime(dates).dropna()
    if len(dt_series) < 2:
        warnings.append("No dates found — seasonal patterns won't be detected.")
        return "unknown", 1, warnings

    dt_series = dt_series.sort_values()
    diffs = dt_series.diff().dropna()
    if len(diffs) == 0:
        return "unknown", 1, warnings

    # Calculate modal gap in days
    days_diffs = diffs.dt.total_seconds() / 86400.0
    # Round to nearest integer day / week / month / quarter
    mode_days = float(pd.Series(days_diffs).mode().iloc[0]) if not pd.Series(days_diffs).mode().empty else float(days_diffs.median())

    if 0.8 <= mode_days <= 1.2:
        freq = "daily"
        seasonal_period = 7
    elif 6.0 <= mode_days <= 8.0:
        freq = "weekly"
        seasonal_period = 52
    elif 27.0 <= mode_days <= 32.0:
        freq = "monthly"
        seasonal_period = 12
    elif 85.0 <= mode_days <= 95.0:
        freq = "quarterly"
        seasonal_period = 4
    elif 350.0 <= mode_days <= 370.0:
        freq = "yearly"
        seasonal_period = 1
    else:
        freq = "custom"
        seasonal_period = 1
        warnings.append(f"Unrecognized spacing (~{mode_days:.1f} days between records). Seasonal period set to 1.")

    return freq, seasonal_period, warnings


def parse_and_validate(
    content: Union[str, bytes],
    filename: Optional[str] = None,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> IngestResult:
    warnings: List[str] = []

    # Handle bytes or string content
    if isinstance(content, bytes):
        if filename and (filename.endswith(".xlsx") or filename.endswith(".xls")):
            df = _read_excel(content)
        else:
            df = _read_table(content.decode("utf-8", errors="replace"))
    elif isinstance(content, str):
        df = _read_table(content)
    else:
        raise IngestError("Unsupported content type provided for ingest.")

    if df is None or df.empty:
        raise IngestError(
            "No data could be read from that input. Check that it contains at least one "
            "column of numbers."
        )

    columns = [str(c) for c in df.columns]
    df.columns = columns

    detected_date_col = date_col if date_col in columns else None
    detected_value_col = value_col if value_col in columns else None

    if len(columns) == 1:
        # One column of numbers, exactly as the paste box advertises.
        detected_date_col = None
        detected_value_col = columns[0]
    else:
        if detected_date_col is None:
            for c in columns:
                if c == detected_value_col:
                    continue
                if _is_date_like_column(df[c]):
                    detected_date_col = c
                    break

        if detected_value_col is None:
            candidates = [c for c in columns if c != detected_date_col]
            best_col, best_score = None, 0.0
            for c in candidates:
                score = float(_clean_numeric_column(df[c]).notna().mean())
                if score > best_score:
                    best_col, best_score = c, score
            if best_col is not None and best_score >= 0.7:
                detected_value_col = best_col
            elif candidates:
                detected_value_col = candidates[-1]

        if detected_value_col is None:
            raise IngestError(
                "Couldn't find a column of numbers. Make sure one column holds the values "
                "you want to forecast."
            )

    # Extract dates
    dates_list = None
    if detected_date_col and detected_date_col in columns:
        parsed_dates = _to_datetime(df[detected_date_col])
        non_na_dates = parsed_dates.dropna()
        dups = non_na_dates[non_na_dates.duplicated()].unique()
        if len(dups) > 0:
            dup_str = ", ".join([str(d)[:10] for d in dups[:5]])
            raise IngestError(
                f"The same date appears more than once: {dup_str}. "
                "Each date needs exactly one value — combine duplicate rows first."
            )
        dates_list = [str(d)[:10] if pd.notna(d) else None for d in parsed_dates]

    # Extract numeric series, tolerating $ , % and (1,200) negatives
    raw_series = df[detected_value_col]
    numeric_series = _clean_numeric_column(raw_series)

    non_numeric_mask = numeric_series.isna() & raw_series.notna()
    offending_count = int(non_numeric_mask.sum())
    if offending_count > 0:
        offending_indices = np.where(non_numeric_mask)[0] + 1
        offending_samples = raw_series.iloc[np.where(non_numeric_mask)[0][:3]].tolist()
        warnings.append(
            f"Skipped {offending_count} row(s) that weren't numbers "
            f"(row {offending_indices[0]}, e.g. \"{offending_samples[0]}\")."
        )

    # Filter out NaNs
    valid_mask = numeric_series.notna()
    clean_series = numeric_series[valid_mask].values.astype(float)

    if dates_list is not None:
        dates_list = [dates_list[i] for i in range(len(dates_list)) if valid_mask.iloc[i]]

        # Oldest-first is required by every model; reorder rather than fail.
        order = pd.Series(_to_datetime(pd.Series(dates_list)))
        if order.notna().all() and not order.is_monotonic_increasing:
            sort_idx = order.argsort().to_numpy()
            clean_series = clean_series[sort_idx]
            dates_list = [dates_list[i] for i in sort_idx]
            warnings.append("Your rows weren't in date order, so they were sorted oldest-first.")

    n = len(clean_series)

    # Validation gates
    if n < 20:
        raise IngestError(
            f"Need at least 20 data points to test any model honestly. You have {n}. "
            "Add more history — monthly data usually needs two to three years."
        )

    if 20 <= n < 50:
        warnings.append(f"With {n} points the accuracy comparison is weak evidence, not proof.")

    if np.all(clean_series == clean_series[0]):
        raise IngestError(
            "Every value is the same, so there is nothing to forecast. "
            "Check that you pasted the right column."
        )

    zero_pct = (clean_series == 0).sum() / n
    if zero_pct > 0.40:
        warnings.append(
            "This looks like intermittent demand. Accuracy numbers here are less reliable, and average-based forecasts may not be the right tool."
        )

    # Frequency and seasonality
    freq, seasonal_period, freq_warnings = infer_frequency_and_seasonality(
        pd.Series(dates_list) if dates_list else None
    )
    warnings.extend(freq_warnings)

    return IngestResult(
        series=clean_series,
        dates=dates_list,
        frequency=freq,
        seasonal_period=seasonal_period,
        warnings=warnings,
        date_column=detected_date_col,
        value_column=detected_value_col,
        available_columns=columns,
    )
