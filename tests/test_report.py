import pytest
from core.report import (
    create_forecast_report,
    export_csv,
    export_pdf,
    generate_verdict_explainer,
    generate_verdict_sentence,
)
from core.backtest import ModelBacktestResult
from core.ingest import IngestResult


def test_verdict_sentence_generation():
    winner = ModelBacktestResult(
        model_name="Theta",
        status="ok",
        mean_mase=0.71,
        coverage=0.93,
    )
    baseline = ModelBacktestResult(
        model_name="Last-value baseline",
        status="ok",
        mean_mase=1.00,
        coverage=0.90,
    )

    verdict = generate_verdict_sentence(winner, baseline)
    assert "Theta was the most accurate." in verdict
    assert "29% less" in verdict
    assert "93% of the time" in verdict


def test_verdict_when_baseline_wins():
    winner = ModelBacktestResult(
        model_name="Last-value baseline",
        status="ok",
        mean_mase=1.00,
        coverage=0.95,
    )

    verdict = generate_verdict_sentence(winner, winner)
    assert "No method beat the simple baseline." in verdict


def test_verdict_explainer_flags_an_unreliable_range():
    """The explainer must tell the user when the range overstates itself."""
    optimistic = ModelBacktestResult(
        model_name="Theta", status="ok", mean_mase=0.6, coverage=0.70
    )
    text = generate_verdict_explainer(optimistic)
    assert "70%" in text
    assert "optimistic" in text

    solid = ModelBacktestResult(
        model_name="Theta", status="ok", mean_mase=0.6, coverage=0.96
    )
    assert "plan against the range" in generate_verdict_explainer(solid)


def test_export_csv_and_pdf():
    report = {
        "winner": "Chronos-2",
        "verdict": "Chronos-2 was the most accurate.",
        "ranking": [
            {"model": "Chronos-2", "status": "ok", "mase": 0.71, "rmse": 400.0, "mae": 300.0, "coverage": 0.93}
        ],
        "forecast": [
            {"step": 1, "date": "2026-01-01", "point": 100.0, "lower": 90.0, "upper": 110.0}
        ],
    }

    csv_data = export_csv(report)
    assert "FORECAST RESULTS" in csv_data
    assert "Chronos-2" in csv_data

    pdf_bytes = export_pdf(report)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")
