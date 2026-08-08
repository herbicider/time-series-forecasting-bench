import io
import math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from core.ingest import IngestResult
from core.backtest import ModelBacktestResult, FoldBoundary


AUTHOR_NAME = "Kevin Xing"
AUTHOR_TAGLINE = "Pharmacy operations & business optimization consulting"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/xiaoang-kevin-xing-0a490359/"
AUTHOR_GITHUB = "https://github.com/herbicider"


def generate_verdict_sentence(
    winner: ModelBacktestResult,
    baseline_result: Optional[ModelBacktestResult],
) -> str:
    if winner is None or winner.status != "ok":
        return "No method completed testing successfully on this data."

    is_baseline_winner = "baseline" in winner.model_name.lower()

    if is_baseline_winner or (baseline_result and winner.mean_mase >= baseline_result.mean_mase):
        return (
            "No method beat the simple baseline. For this data, repeating the last known value is as good as anything "
            "more complicated. That usually means the series is close to random, or there is not enough history yet."
        )

    # Calculate percentage improvement over baseline
    base_mase = baseline_result.mean_mase if (baseline_result and baseline_result.mean_mase > 0) else 1.0
    pct_better = int(round((1.0 - (winner.mean_mase / base_mase)) * 100))
    pct_better = max(1, pct_better)

    cov_pct = int(round(winner.coverage * 100))

    s1 = f"{winner.model_name} was the most accurate."
    s2 = f"It missed by {pct_better}% less than simply repeating the last known value."
    s3 = f"Its predicted range was right {cov_pct}% of the time on history it had never seen."

    return f"{s1} {s2} {s3}"


def generate_verdict_explainer(winner: Optional[ModelBacktestResult]) -> str:
    """One plain sentence on what to actually do with the verdict."""
    if winner is None or winner.status != "ok":
        return (
            "Try adding more history, or check that the column you pasted holds the "
            "numbers you meant to forecast."
        )

    mase = winner.mean_mase
    if mase < 0.75:
        quality = "This is a strong result — the pattern in your data is learnable."
    elif mase < 1.0:
        quality = "This is a usable result, but keep the range in mind when you plan."
    else:
        quality = (
            "Nothing beat the naive guess, so treat any single number here with caution."
        )

    coverage_pct = int(round(winner.coverage * 100))
    if coverage_pct >= 90:
        reliability = (
            f"The range held up {coverage_pct}% of the time, so plan against the range "
            "rather than the single number."
        )
    else:
        reliability = (
            f"The range only held {coverage_pct}% of the time, so treat it as optimistic "
            "and leave yourself extra room."
        )

    return f"{quality} {reliability}"


def create_forecast_report(
    ingest_res: IngestResult,
    horizon: int,
    backtest_results: List[ModelBacktestResult],
    boundaries: List[FoldBoundary],
    extra_warnings: List[str],
) -> Dict[str, Any]:
    warnings = list(ingest_res.warnings) + extra_warnings

    # Find winner and baseline
    ok_results = [r for r in backtest_results if r.status == "ok"]
    baseline = next((r for r in backtest_results if "Last-value baseline" in r.model_name), None)
    winner = ok_results[0] if ok_results else None

    winner_name = winner.model_name if winner else "None"
    verdict = generate_verdict_sentence(winner, baseline)

    # Generate future dates if date column existed
    future_dates = []
    if ingest_res.dates and len(ingest_res.dates) > 0:
        try:
            last_date = pd.to_datetime(ingest_res.dates[-1])
            freq_str = {
                "daily": "D",
                "weekly": "W",
                "monthly": "MS",
                "quarterly": "QS",
                "yearly": "YS",
            }.get(ingest_res.frequency, "D")
            date_range = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_str)[1:]
            future_dates = [d.strftime("%Y-%m-%d") for d in date_range]
        except Exception:
            future_dates = [f"Step {h}" for h in range(1, horizon + 1)]
    else:
        future_dates = [f"Step {h}" for h in range(1, horizon + 1)]

    # Forecast table for winner
    forecast_rows = []
    if winner and winner.final_forecast:
        fc = winner.final_forecast
        c_low = winner.calibrated_lower if winner.calibrated_lower is not None else fc.lower
        c_up = winner.calibrated_upper if winner.calibrated_upper is not None else fc.upper

        for h in range(horizon):
            row = {
                "step": h + 1,
                "date": future_dates[h] if h < len(future_dates) else f"Step {h+1}",
                "point": round(float(fc.point[h]), 2),
                "lower": round(float(c_low[h]), 2),
                "upper": round(float(c_up[h]), 2),
                "raw_lower": round(float(fc.lower[h]), 2),
                "raw_upper": round(float(fc.upper[h]), 2),
            }
            forecast_rows.append(row)

    # Series metadata
    start_date = ingest_res.dates[0] if ingest_res.dates else "1"
    end_date = ingest_res.dates[-1] if ingest_res.dates else str(len(ingest_res.series))

    min_train = boundaries[0].train_end - boundaries[0].train_start if boundaries else 16

    ranking_list = [r.to_dict() for r in backtest_results]

    report = {
        "series": {
            "n": len(ingest_res.series),
            "frequency": ingest_res.frequency,
            "seasonal_period": ingest_res.seasonal_period,
            "start": start_date,
            "end": end_date,
            "history": ingest_res.series.tolist(),
            "history_dates": ingest_res.dates or [str(i+1) for i in range(len(ingest_res.series))],
        },
        "backtest": {
            "folds": len(boundaries),
            "horizon": horizon,
            "min_train": min_train,
            "boundaries": [
                {
                    "fold": b.fold,
                    "train_start": b.train_start,
                    "train_end": b.train_end,
                    "test_start": b.test_start,
                    "test_end": b.test_end,
                }
                for b in boundaries
            ],
        },
        "ranking": ranking_list,
        "winner": winner_name,
        "verdict": verdict,
        "verdict_explainer": generate_verdict_explainer(winner),
        "forecast": forecast_rows,
        "warnings": warnings,
        "author": {
            "name": AUTHOR_NAME,
            "tagline": AUTHOR_TAGLINE,
            "linkedin": AUTHOR_LINKEDIN,
            "github": AUTHOR_GITHUB,
        },
    }

    return report


def export_csv(report: Dict[str, Any]) -> str:
    output = io.StringIO()

    output.write("FORECAST RESULTS\n")
    output.write(f"Winner,{report.get('winner', '')}\n")
    output.write(f"Verdict,\"{report.get('verdict', '')}\"\n\n")

    output.write("MODEL RANKING\n")
    ranking_df = pd.DataFrame(report.get("ranking", []))
    ranking_df.to_csv(output, index=False)
    output.write("\n")

    output.write("FORECAST DETAILS\n")
    forecast_df = pd.DataFrame(report.get("forecast", []))
    forecast_df.to_csv(output, index=False)

    return output.getvalue()


def export_pdf(report: Dict[str, Any]) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#151C24"),
        spaceAfter=12,
    )
    verdict_style = ParagraphStyle(
        "ReportVerdict",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#7C2D3A"),
        spaceAfter=18,
    )
    section_style = ParagraphStyle(
        "ReportSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#151C24"),
        spaceBefore=10,
        spaceAfter=6,
    )

    byline_style = ParagraphStyle(
        "ReportByline",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#59636E"),
        spaceAfter=14,
    )
    footer_style = ParagraphStyle(
        "ReportFooter",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#59636E"),
        spaceBefore=18,
    )

    elements = []
    elements.append(Paragraph("Time Series Forecasting Report", title_style))
    elements.append(
        Paragraph(
            f"Generated by Time Series Forecasting Bench &middot; built by {AUTHOR_NAME}",
            byline_style,
        )
    )
    elements.append(Paragraph(report.get("verdict", ""), verdict_style))
    explainer = report.get("verdict_explainer")
    if explainer:
        elements.append(Paragraph(explainer, byline_style))

    # Ranking Table
    elements.append(Paragraph("Model Evaluation Ranking", section_style))
    ranking_data = [["Model", "Status", "MASE", "RMSE", "MAE", "Coverage"]]
    def _num(value):
        # Explicit None check: a legitimate metric of exactly 0.0 is falsy and
        # used to render as "-", which reads as "this model did not run".
        return "-" if value is None else str(value)

    for r in report.get("ranking", []):
        ranking_data.append(
            [
                r.get("model", ""),
                r.get("status", ""),
                _num(r.get("mase")),
                _num(r.get("rmse")),
                _num(r.get("mae")),
                f"{int(round((r.get('coverage') or 0) * 100))}%",
            ]
        )

    t_rank = Table(ranking_data, colWidths=[130, 80, 70, 70, 70, 70])
    t_rank.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F5F3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#151C24")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(t_rank)
    elements.append(Spacer(1, 14))

    # Forecast Table
    elements.append(Paragraph(f"Forecast ({report.get('winner', '')})", section_style))
    forecast_data = [["Step", "Date", "Point", "Lower (95%)", "Upper (95%)"]]
    for f in report.get("forecast", []):
        forecast_data.append(
            [
                str(f.get("step", "")),
                str(f.get("date", "")),
                f"{f.get('point', 0):,.2f}",
                f"{f.get('lower', 0):,.2f}",
                f"{f.get('upper', 0):,.2f}",
            ]
        )

    t_fc = Table(forecast_data, colWidths=[50, 110, 110, 110, 110])
    t_fc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F5F3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#151C24")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(t_fc)

    elements.append(
        Paragraph(
            f"<b>{AUTHOR_NAME}</b> &mdash; {AUTHOR_TAGLINE}.<br/>"
            f'Questions about what this means for your pharmacy? '
            f'<a href="{AUTHOR_LINKEDIN}" color="#7C2D3A">Message me on LinkedIn</a> '
            f'&middot; <a href="{AUTHOR_GITHUB}" color="#7C2D3A">github.com/herbicider</a>',
            footer_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
