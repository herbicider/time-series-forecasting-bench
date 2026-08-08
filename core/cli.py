import argparse
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest import parse_and_validate, IngestError
from core.backtest import run_backtest
from core.report import create_forecast_report, export_csv, export_pdf



def run_cli(args=None):
    parser = argparse.ArgumentParser(
        prog="forecasting-bench",
        description="Time Series Forecasting Bench — forecasting with honest backtesting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run forecast on a dataset file or raw string.")
    run_parser.add_argument("input_path", type=str, help="Path to CSV/XLSX file or inline text string")
    run_parser.add_argument("--horizon", "-n", type=int, default=6, help="How many periods ahead to forecast (default: 6)")
    run_parser.add_argument("--date-col", type=str, default=None, help="Name of date column")
    run_parser.add_argument("--value-col", type=str, default=None, help="Name of value column")
    run_parser.add_argument("--export-csv", type=str, default=None, help="Path to export CSV report")
    run_parser.add_argument("--export-pdf", type=str, default=None, help="Path to export PDF report")
    run_parser.add_argument("--json", action="store_true", help="Output raw JSON report")

    parsed = parser.parse_args(args)

    if parsed.command == "run":
        target = parsed.input_path
        content = None
        filename = None

        if os.path.exists(target):
            filename = target
            with open(target, "rb") as f:
                content = f.read()
        else:
            content = target

        try:
            ingest_res = parse_and_validate(
                content=content,
                filename=filename,
                date_col=parsed.date_col,
                value_col=parsed.value_col,
            )
        except IngestError as e:
            print(f"Error reading dataset: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Dataset ingested: {len(ingest_res.series)} points ({ingest_res.frequency} frequency).")
        print("Running backtest and model bake-off...")

        backtest_results, boundaries, extra_warnings = run_backtest(
            y=ingest_res.series,
            horizon=parsed.horizon,
            seasonal_period=ingest_res.seasonal_period,
        )

        report = create_forecast_report(
            ingest_res=ingest_res,
            horizon=parsed.horizon,
            backtest_results=backtest_results,
            boundaries=boundaries,
            extra_warnings=extra_warnings,
        )

        if parsed.json:
            print(json.dumps(report, indent=2))
            return

        # Terminal text display
        print("\n" + "=" * 60)
        print("TIME SERIES FORECASTING BENCH")
        print("=" * 60)
        print(f"\nVERDICT:\n{report['verdict']}\n")

        if report.get("warnings"):
            print("WARNINGS:")
            for w in report["warnings"]:
                print(f" - {w}")
            print()

        print("MODEL RANKING:")
        width = max([len(r["model"]) for r in report["ranking"]] + [22])
        print(f"{'Method':<{width}} {'Status':<10} {'MASE':<8} {'RMSE':<12} {'MAE':<12} {'Coverage':<8}")
        print("-" * (width + 54))
        for r in report["ranking"]:
            # Explicit None checks: a legitimate metric of exactly 0.0 is falsy
            # and used to render as "-", which reads as "did not run".
            mase_str = "-" if r.get("mase") is None else f"{r['mase']:.3f}"
            rmse_str = "-" if r.get("rmse") is None else f"{r['rmse']:,.2f}"
            mae_str = "-" if r.get("mae") is None else f"{r['mae']:,.2f}"
            cov_str = f"{int(round((r.get('coverage') or 0) * 100))}%"
            print(f"{r['model']:<{width}} {r['status']:<10} {mase_str:<8} {rmse_str:<12} {mae_str:<12} {cov_str:<8}")

        print(f"\nFORECAST ({report['winner']}):")
        print(f"{'Step':<6} {'Date':<14} {'Prediction':>14} {'Low end':>14} {'High end':>14}")
        print("-" * 66)
        for f in report["forecast"]:
            print(f"{f['step']:<6} {f['date']:<14} {f['point']:>14,.2f} {f['lower']:>14,.2f} {f['upper']:>14,.2f}")

        author = report.get("author") or {}
        if author:
            print(f"\nBuilt by {author.get('name')} — {author.get('tagline')}.")
            print(f"  {author.get('linkedin')}")

        if parsed.export_csv:
            csv_data = export_csv(report)
            with open(parsed.export_csv, "w") as f_csv:
                f_csv.write(csv_data)
            print(f"\nCSV report exported to {parsed.export_csv}")

        if parsed.export_pdf:
            pdf_data = export_pdf(report)
            with open(parsed.export_pdf, "wb") as f_pdf:
                f_pdf.write(pdf_data)
            print(f"\nPDF report exported to {parsed.export_pdf}")


if __name__ == "__main__":
    run_cli()
