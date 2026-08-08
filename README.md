# Time Series Forecasting Bench

**Forecast your pharmacy's numbers — and find out which forecast you can actually trust.**

Paste a column of numbers. The app tests seven forecasting methods against your own
history, scores each one on data it was never allowed to see, and tells you in plain
English which to use and how much to rely on it.

> **Your data never leaves your computer.** No server, no account, no telemetry, no
> internet connection required.

---

## Download

No installation. No admin rights. Nothing written to Program Files or the registry.

| Platform | Download | What to do |
|---|---|---|
| **Windows** | ~110 MB zip | Unzip anywhere, double-click **ForecastingBench.exe** |
| **macOS** | ~92 MB zip | Unzip, drag to Applications. **First launch:** right-click the app → *Open* (it is unsigned, so a normal double-click is blocked once) |

Grab the latest from
**[Releases](https://github.com/herbicider/time-series-forecasting-bench/releases)**.

---

## How to use it

**1. Add your numbers.** Any of these three work:

```
Just the numbers          Date + number             A spreadsheet
5200                      2023-01-01, 5200          Drag a .csv or .xlsx
5400                      2023-02-01, 5400          onto the window
5800                      2023-03-01, 5800
```

Oldest first. Dollar signs and commas are fine — `$12,400` works. Adding dates lets
the app detect seasonal patterns. You need at least 20 data points; 30+ is much more
reliable.

**2. Choose how far ahead** you want to forecast.

**3. Read the verdict.** You get a ranked table, a chart with a prediction range, and
a one-sentence answer in plain English.

---

## What it actually does

Most forecasting tools show you one number and no way to judge it. This one runs a
**rolling-origin backtest**: it repeatedly hides the last few periods of your history,
forecasts them, and compares against what really happened. Every method is scored the
same way, on the same windows.

The prediction range is then calibrated by **split conformal** — and, importantly, the
reliability percentage you see is measured on windows that were *not* used to calibrate
it. That is why a method can win on accuracy and still show a range reliability well
below 95%: the number is honest rather than circular.

### Methods compared

| Method | Notes |
|---|---|
| Last-value baseline | Seasonal naive — the honest thing to beat |
| Trend baseline | Straight-line drift |
| ARIMA | `statsforecast` AutoARIMA |
| Exponential Smoothing (ETS) | `statsforecast` AutoETS |
| Theta | Winner of the M3 competition |
| Smoothed Trend (built-in) | Holt-style level + trend |
| Seasonal Weighted Average (built-in) | Weighted recent average with a seasonal factor |

**On the AI Edition.** A separate build adds Google TimesFM 2.5 and Amazon Chronos-2.
It is a much larger download and fetches ~1.3 GB of model weights on first use, behind
a progress screen. It is worth it mainly for long or unusual histories — for monthly
pharmacy revenue, ARIMA and ETS are usually competitive.

Methods are **never mislabelled**: a row says "Google TimesFM 2.5" only if that model
genuinely produced the numbers. The built-in heuristics are always marked `(built-in)`.

---

## Run from source

```bash
git clone https://github.com/herbicider/time-series-forecasting-bench.git
cd time-series-forecasting-bench

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python shell/app.py               # desktop app
python core/cli.py run samples/monthly_revenue.csv   # headless
```

Optional AI edition:

```bash
pip install -r requirements-ai.txt
```

### Build a portable app

```bash
pip install pyinstaller
FB_EDITION=standard pyinstaller packaging/forecasting_bench.spec --noconfirm
```

Output lands in `dist/`. Windows produces a portable folder; macOS produces a `.app`.

### Tests

```bash
python -m pytest tests/ -q
```

---

## Repo layout

```
core/                    Domain logic — no UI, no web framework
  ingest.py              Paste/CSV/XLSX parsing, format sniffing, validation
  models/                Forecasters + the capability manager
  backtest.py            Rolling-origin evaluation
  conformal.py           Split-conformal interval calibration
  report.py              Verdict generation, CSV/PDF export
service/main.py          Local FastAPI service (job queue, progress, exports)
ui/                      Static HTML/CSS/JS + vendored ECharts and fonts
shell/app.py             pywebview desktop window
packaging/               PyInstaller spec (both platforms, both editions)
samples/                 Pharmacy example datasets
tests/                   pytest suite
```

---

## About

Built by **Kevin Xing** — pharmacy operations & business optimization consulting.

If you'd like to talk through what your numbers mean for your pharmacy — staffing,
purchasing, inventory, or growth — send me a message on
**[LinkedIn](https://www.linkedin.com/in/xiaoang-kevin-xing-0a490359/)**.

- LinkedIn: <https://www.linkedin.com/in/xiaoang-kevin-xing-0a490359/>
- GitHub: <https://github.com/herbicider>

---

## License

[Apache 2.0](LICENSE). Third-party attributions in [NOTICE](NOTICE).
