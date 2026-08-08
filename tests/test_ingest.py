import pytest
import numpy as np
from core.ingest import parse_and_validate, IngestError, infer_frequency_and_seasonality


def _numbers(n=30, base=100, step=3):
    return [base + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def test_ingest_refuse_less_than_20_points():
    csv_data = "date,value\n" + "\n".join([f"2023-01-{i+1:02d},{100+i}" for i in range(15)])
    with pytest.raises(IngestError) as excinfo:
        parse_and_validate(csv_data)
    assert "Need at least 20 data points" in str(excinfo.value)


def test_ingest_warn_20_to_50_points():
    csv_data = "date,value\n" + "\n".join([f"2023-01-{i+1:02d},{100+i}" for i in range(30)])
    res = parse_and_validate(csv_data)
    assert len(res.series) == 30
    assert any("weak evidence" in w for w in res.warnings)


def test_ingest_refuse_all_identical():
    csv_data = "date,value\n" + "\n".join([f"2023-01-{i+1:02d},100" for i in range(25)])
    with pytest.raises(IngestError) as excinfo:
        parse_and_validate(csv_data)
    assert "Every value is the same" in str(excinfo.value)


def test_ingest_monthly_revenue_sample():
    with open("samples/monthly_revenue.csv", "r") as f:
        content = f.read()
    res = parse_and_validate(content)
    assert len(res.series) == 48
    assert res.frequency == "monthly"
    assert res.seasonal_period == 12


# ---------------------------------------------------------------------------
# Accepted paste formats.
#
# Regression guard: every one of these used to raise IngestError even though
# the paste box advertised them. A bare number must never be read as a date.
# ---------------------------------------------------------------------------

def test_bare_numbers_one_per_line():
    res = parse_and_validate("\n".join(str(v) for v in _numbers()))
    assert len(res.series) == 30
    assert res.dates is None
    assert res.series[0] == 100.0
    assert res.series[-1] == 187.0


def test_bare_numbers_with_header():
    res = parse_and_validate("value\n" + "\n".join(str(v) for v in _numbers()))
    assert len(res.series) == 30
    assert res.dates is None


def test_headerless_date_value_pairs():
    text = "\n".join(f"2023-{(i//28)+1:02d}-{(i%28)+1:02d},{100+i}" for i in range(30))
    res = parse_and_validate(text)
    assert len(res.series) == 30
    assert res.dates is not None
    assert res.series[0] == 100.0


def test_excel_tab_delimited_paste():
    text = "\n".join(f"2023-{(i//28)+1:02d}-{(i%28)+1:02d}\t{100+i}" for i in range(30))
    res = parse_and_validate(text)
    assert len(res.series) == 30
    assert res.dates is not None


def test_currency_and_thousands_separators():
    res = parse_and_validate("\n".join(f"${100+i*3:,}.00" for i in range(30)))
    assert len(res.series) == 30
    assert res.series[0] == 100.0
    assert res.series[-1] == 187.0


def test_parenthesized_negatives():
    values = [(-1) ** i * (50 + i) for i in range(30)]
    text = "\n".join((f"({abs(v)})" if v < 0 else str(v)) for v in values)
    res = parse_and_validate(text)
    assert list(res.series) == [float(v) for v in values]


def test_us_slash_dates():
    text = "\n".join(f"{(i%12)+1:02d}/15/202{i//12}, {100+i}" for i in range(30))
    res = parse_and_validate(text)
    assert len(res.series) == 30
    assert res.frequency == "monthly"


def test_quarterly_labels():
    text = "date,value\n" + "\n".join(f"{2015+i//4}-Q{(i%4)+1},{100+i}" for i in range(30))
    res = parse_and_validate(text)
    assert res.frequency == "quarterly"
    assert res.seasonal_period == 4


def test_crlf_and_blank_lines_are_tolerated():
    text = "\r\n".join(str(v) for v in _numbers()) + "\r\n\r\n"
    res = parse_and_validate(text)
    assert len(res.series) == 30


def test_surrounding_whitespace_is_tolerated():
    res = parse_and_validate("\n".join(f"  {v}  " for v in _numbers()))
    assert len(res.series) == 30


def test_rows_out_of_date_order_are_sorted():
    text = "date,value\n" + "\n".join(
        f"2023-{(i//28)+1:02d}-{(i%28)+1:02d},{100+i}" for i in reversed(range(30))
    )
    res = parse_and_validate(text)
    assert res.series[0] == 100.0
    assert res.series[-1] == 129.0
    assert any("date order" in w for w in res.warnings)


def test_non_numeric_rows_are_skipped_with_warning():
    text = "value\n" + "\n".join(("N/A" if i == 5 else str(100 + i * 3)) for i in range(32))
    res = parse_and_validate(text)
    assert len(res.series) == 31
    assert any("weren't numbers" in w for w in res.warnings)


def test_duplicate_dates_are_rejected_with_actionable_message():
    text = "date,value\n" + "\n".join(f"2023-01-01,{100+i}" for i in range(25))
    with pytest.raises(IngestError) as excinfo:
        parse_and_validate(text)
    assert "more than once" in str(excinfo.value)


def test_empty_input_is_rejected():
    with pytest.raises(IngestError):
        parse_and_validate("   \n  \n")


@pytest.mark.parametrize(
    "path,n,freq",
    [
        ("samples/monthly_revenue.csv", 48, "monthly"),
        ("samples/weekly_inventory_units.csv", 104, "weekly"),
        ("samples/daily_rx_count.csv", 180, "daily"),
        ("samples/active_patients.csv", 42, "monthly"),
    ],
)
def test_bundled_samples_all_parse(path, n, freq):
    with open(path, "r") as f:
        res = parse_and_validate(f.read())
    assert len(res.series) == n
    assert res.frequency == freq
