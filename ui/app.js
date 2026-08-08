document.addEventListener("DOMContentLoaded", () => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // Input
  const dataInput = $("data-input");
  const dropZone = $("drop-zone");
  const filePicker = $("file-picker");
  const fileNameDisplay = $("file-name-display");
  const btnClearFile = $("btn-clear-file");
  const inputSummary = $("input-summary");
  const horizonInput = $("horizon-input");
  const horizonCaption = $("horizon-caption");
  const aiPanel = $("ai-panel");
  const btnRun = $("btn-run");

  // Running
  const progressFill = $("progress-fill");
  const progressBar = $("progress-bar");
  const progressMessage = $("progress-message");
  const progressPct = $("progress-pct");
  const stageList = $("stage-list");
  const btnCancel = $("btn-cancel");

  // Download
  const dlFill = $("dl-progress-fill");
  const dlBar = $("dl-progress-bar");
  const dlMessage = $("dl-message");
  const dlPct = $("dl-pct");
  const dlError = $("dl-error");

  let currentReport = null;
  let chartInstance = null;
  let selectedFile = null;
  let activeJobId = null;
  let pollTimer = null;
  let capabilities = null;
  let lastFrequency = "unknown";

  // -----------------------------------------------------------------------
  // External links
  //
  // pywebview navigates the app window itself for a normal anchor, which
  // strands the user inside the desktop app with no back button. The Python
  // side exposes open_url(); fall back to a normal new tab in a browser.
  // -----------------------------------------------------------------------
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-external]");
    if (!link) return;
    event.preventDefault();
    const url = link.getAttribute("href");
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_url) {
      window.pywebview.api.open_url(url);
    } else {
      window.open(url, "_blank", "noopener");
    }
  });

  // -----------------------------------------------------------------------
  // Horizon stepper
  // -----------------------------------------------------------------------
  const FREQ_UNITS = {
    daily: ["day", "days"],
    weekly: ["week", "weeks"],
    monthly: ["month", "months"],
    quarterly: ["quarter", "quarters"],
    yearly: ["year", "years"],
  };

  function updateHorizonCaption() {
    const value = parseInt(horizonInput.value, 10) || 1;
    const unit = FREQ_UNITS[lastFrequency] || ["period", "periods"];
    horizonCaption.textContent = `${value} ${value === 1 ? unit[0] : unit[1]} ahead`;
  }

  function stepHorizon(delta) {
    const current = parseInt(horizonInput.value, 10) || 6;
    horizonInput.value = Math.max(1, Math.min(24, current + delta));
    updateHorizonCaption();
  }

  $("btn-horizon-dec").addEventListener("click", () => stepHorizon(-1));
  $("btn-horizon-inc").addEventListener("click", () => stepHorizon(1));
  horizonInput.addEventListener("input", updateHorizonCaption);
  updateHorizonCaption();

  // -----------------------------------------------------------------------
  // Live input summary — immediate feedback that the paste was understood
  // -----------------------------------------------------------------------
  function describeInput() {
    const text = dataInput.value.trim();
    if (selectedFile) {
      inputSummary.textContent = `Using file: ${selectedFile.name}`;
      inputSummary.className = "input-summary";
      return;
    }
    if (!text) {
      inputSummary.className = "input-summary hidden";
      return;
    }
    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    // Look at the first few rows, not just row 1 — row 1 is often a header
    // like "date,value", which contains no date at all.
    const DATE_HINT = /\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\bQ[1-4]\b/i;
    const looksDated = lines.slice(0, 5).some((l) => DATE_HINT.test(l));
    inputSummary.className = "input-summary";
    if (lines.length < 20) {
      inputSummary.classList.add("input-summary--warn");
      inputSummary.textContent =
        `${lines.length} row${lines.length === 1 ? "" : "s"} so far — you need at least 20.`;
    } else {
      inputSummary.textContent =
        `${lines.length} rows detected` +
        (looksDated ? ", with dates — seasonal patterns will be checked." : ". No dates, so seasonality will be skipped.");
    }
  }
  dataInput.addEventListener("input", describeInput);

  // -----------------------------------------------------------------------
  // Sample data. These now come from the real files under /samples, which the
  // service mounts explicitly — the old fetch always 404'd and silently served
  // a hardcoded ramp instead.
  // -----------------------------------------------------------------------
  document.querySelectorAll(".chip-btn").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const key = chip.dataset.sample;
      clearError();
      try {
        const resp = await fetch(`/samples/${key}.csv`);
        if (!resp.ok) throw new Error(`Sample not available (${resp.status})`);
        dataInput.value = (await resp.text()).trim();
        clearFile();
        describeInput();
        document.querySelectorAll(".chip-btn").forEach((c) => c.classList.remove("chip-btn--active"));
        chip.classList.add("chip-btn--active");
      } catch (err) {
        showError(`Could not load that example: ${err.message}`);
      }
    });
  });

  // -----------------------------------------------------------------------
  // File selection
  // -----------------------------------------------------------------------
  function setFile(file) {
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    fileNameDisplay.classList.add("file-name--set");
    btnClearFile.classList.remove("hidden");
    dataInput.value = "";
    document.querySelectorAll(".chip-btn").forEach((c) => c.classList.remove("chip-btn--active"));
    describeInput();
  }

  function clearFile() {
    selectedFile = null;
    filePicker.value = "";
    fileNameDisplay.textContent = "No file selected";
    fileNameDisplay.classList.remove("file-name--set");
    btnClearFile.classList.add("hidden");
    describeInput();
  }

  filePicker.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length) setFile(e.target.files[0]);
  });
  btnClearFile.addEventListener("click", clearFile);

  ["dragenter", "dragover"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    })
  );
  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
      clearError();
    }
  });

  // -----------------------------------------------------------------------
  // Capabilities — what this build can run
  // -----------------------------------------------------------------------
  async function loadCapabilities() {
    try {
      const resp = await fetch("/api/capabilities");
      if (!resp.ok) return;
      capabilities = await resp.json();
      renderAiPanel();
    } catch (_) {
      /* Standard edition works fine without this. */
    }
  }

  function renderAiPanel() {
    if (!capabilities) return;
    aiPanel.innerHTML = "";
    aiPanel.classList.remove("hidden");

    const title = document.createElement("h4");
    const body = document.createElement("p");

    if (!capabilities.ai_edition) {
      title.textContent = "Advanced AI forecasters";
      body.textContent =
        "This is the Standard edition. It runs seven proven forecasting methods " +
        "with no download and no internet. The AI Edition adds Google TimesFM and " +
        "Amazon Chronos — a much larger download, worth it mainly for long or " +
        "unusual histories.";
      aiPanel.append(title, body);
      return;
    }

    if (capabilities.needs_download) {
      title.textContent = "Advanced AI forecasters are ready to download";
      body.textContent =
        `Google TimesFM and Amazon Chronos need a one-time download of about ` +
        `${capabilities.pending_mb} MB. Until then, the seven built-in methods are used.`;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-secondary";
      btn.textContent = `Download AI models (~${capabilities.pending_mb} MB, one time)`;
      btn.addEventListener("click", startModelDownload);
      aiPanel.append(title, body, btn);
      return;
    }

    title.textContent = "Advanced AI forecasters are ready";
    body.textContent = "Google TimesFM and Amazon Chronos will be included in the comparison.";
    aiPanel.append(title, body);
  }

  // -----------------------------------------------------------------------
  // Model download
  // -----------------------------------------------------------------------
  async function startModelDownload() {
    dlError.classList.add("hidden");
    setProgress(dlFill, dlBar, dlPct, 0);
    dlMessage.textContent = "Starting download…";
    switchState("download");
    try {
      await fetch("/api/models/download", { method: "POST" });
    } catch (err) {
      dlError.textContent = `Could not start the download: ${err.message}`;
      dlError.classList.remove("hidden");
      return;
    }
    pollDownload();
  }

  function pollDownload() {
    const tick = async () => {
      try {
        const status = await (await fetch("/api/models/download")).json();
        setProgress(dlFill, dlBar, dlPct, status.pct || 0);
        dlMessage.textContent = status.message || "Working…";
        if (status.state === "done") {
          await loadCapabilities();
          switchState("input");
          return;
        }
        if (status.state === "error") {
          dlError.textContent =
            `${status.error || "The download failed."} — the built-in methods still work, ` +
            `so you can go back and run a forecast now.`;
          dlError.classList.remove("hidden");
          return;
        }
      } catch (err) {
        dlError.textContent = `Lost contact with the download: ${err.message}`;
        dlError.classList.remove("hidden");
        return;
      }
      setTimeout(tick, 700);
    };
    tick();
  }

  $("btn-dl-back").addEventListener("click", () => switchState("input"));

  // -----------------------------------------------------------------------
  // Run the forecast
  // -----------------------------------------------------------------------
  function setProgress(fillEl, barEl, pctEl, pct) {
    const clamped = Math.max(0, Math.min(100, pct));
    fillEl.style.width = `${clamped}%`;
    pctEl.textContent = `${Math.round(clamped)}%`;
    barEl.setAttribute("aria-valuenow", String(Math.round(clamped)));
  }

  function resetStages() {
    stageList.innerHTML = "";
  }

  function markStage(label) {
    const existing = Array.from(stageList.children).find((li) => li.dataset.stage === label);
    if (existing) return;
    Array.from(stageList.children).forEach((li) => {
      li.className = "stage-item completed";
      li.querySelector(".stage-icon").textContent = "✓";
    });
    const li = document.createElement("li");
    li.className = "stage-item active";
    li.dataset.stage = label;
    const icon = document.createElement("span");
    icon.className = "stage-icon";
    icon.textContent = "●";
    const text = document.createElement("span");
    text.textContent = label;
    li.append(icon, text);
    stageList.appendChild(li);
  }

  btnRun.addEventListener("click", async () => {
    const rawData = dataInput.value.trim();
    if (!selectedFile && !rawData) {
      showError("Please paste your numbers, or choose a file first.");
      return;
    }

    clearError();
    resetStages();
    setProgress(progressFill, progressBar, progressPct, 0);
    progressMessage.textContent = "Reading and checking your data…";
    switchState("running");

    const horizon = parseInt(horizonInput.value, 10) || 6;

    try {
      let resp;
      if (selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("horizon", horizon);
        resp = await fetch("/api/forecast/file", { method: "POST", body: formData });
      } else {
        resp = await fetch("/api/forecast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data: rawData, horizon }),
        });
      }
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || "Could not start the forecast.");
      }
      activeJobId = (await resp.json()).job_id;
      pollJob();
    } catch (err) {
      switchState("input");
      showError(err.message);
    }
  });

  function pollJob() {
    const tick = async () => {
      if (!activeJobId) return;
      try {
        const resp = await fetch(`/api/job/${activeJobId}`);
        if (!resp.ok) throw new Error("That forecast run expired. Please try again.");
        const status = await resp.json();

        setProgress(progressFill, progressBar, progressPct, status.pct || 0);
        if (status.message) progressMessage.textContent = status.message;
        if (status.stage) markStage(status.stage);

        if (status.state === "done") {
          activeJobId = null;
          currentReport = status.report;
          renderResults(currentReport);
          switchState("results");
          return;
        }
        if (status.state === "error") {
          activeJobId = null;
          switchState("input");
          showError(status.error || "The forecast failed.");
          return;
        }
        if (status.state === "cancelled") {
          activeJobId = null;
          switchState("input");
          return;
        }
      } catch (err) {
        activeJobId = null;
        switchState("input");
        showError(err.message);
        return;
      }
      pollTimer = setTimeout(tick, 300);
    };
    tick();
  }

  btnCancel.addEventListener("click", async () => {
    if (pollTimer) clearTimeout(pollTimer);
    const jobId = activeJobId;
    activeJobId = null;
    switchState("input");
    if (jobId) {
      try {
        await fetch(`/api/job/${jobId}/cancel`, { method: "POST" });
      } catch (_) {
        /* The window is already back on the input screen. */
      }
    }
  });

  $("btn-new-forecast").addEventListener("click", () => switchState("input"));

  // -----------------------------------------------------------------------
  // Results
  // -----------------------------------------------------------------------
  function renderResults(report) {
    lastFrequency = (report.series && report.series.frequency) || "unknown";
    updateHorizonCaption();

    $("verdict-text").textContent = report.verdict || "";
    $("verdict-explainer").textContent = report.verdict_explainer || "";

    const winner = (report.ranking || []).find((r) => r.model === report.winner);
    const covPct = Math.round(((winner && winner.coverage) || 0) * 100);
    const stripFill = $("strip-fill");
    stripFill.style.width = `${Math.min(100, covPct)}%`;
    stripFill.className = covPct < 90 ? "strip-fill caution" : "strip-fill";
    $("strip-legend").textContent = `The range was right ${covPct}% of the time (95% is the target)`;

    // Warnings — textContent, never innerHTML: these strings can embed raw
    // cell values from the user's own file.
    const warningsCard = $("warnings-card");
    const warningsList = $("warnings-list");
    warningsList.innerHTML = "";
    const warnings = report.warnings || [];
    if (warnings.length) {
      warnings.forEach((w) => {
        const li = document.createElement("li");
        li.textContent = w;
        warningsList.appendChild(li);
      });
      warningsCard.classList.remove("hidden");
    } else {
      warningsCard.classList.add("hidden");
    }

    renderRanking(report);
    renderForecastTable(report);
    renderDiagnostics(report);
    renderChart(report);
  }

  function renderRanking(report) {
    const body = $("ranking-body");
    body.innerHTML = "";
    let rank = 0;

    (report.ranking || []).forEach((row) => {
      const tr = document.createElement("tr");
      const ok = row.status === "ok";
      if (ok) rank += 1;
      const isWinner = ok && row.model === report.winner;
      if (isWinner) tr.className = "winner-row";

      const cells = [
        ok ? String(rank) : "—",
        row.model,
        ok ? (isWinner ? "Winner" : "Tested") : row.status,
        row.mase === null || row.mase === undefined ? "—" : row.mase.toFixed(3),
        row.coverage === null || row.coverage === undefined
          ? "—"
          : `${Math.round(row.coverage * 100)}%`,
        row.rmse === null || row.rmse === undefined ? "—" : fmt(row.rmse),
        row.mae === null || row.mae === undefined ? "—" : fmt(row.mae),
      ];

      cells.forEach((value, i) => {
        const td = document.createElement("td");
        if (i === 2 && isWinner) {
          const badge = document.createElement("span");
          badge.className = "winner-badge";
          badge.textContent = "Winner";
          td.appendChild(badge);
        } else {
          td.textContent = value;
        }
        if (i === 1 && !ok && row.error_reason) td.title = row.error_reason;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  function renderForecastTable(report) {
    const body = $("forecast-body");
    body.innerHTML = "";
    (report.forecast || []).forEach((row) => {
      const tr = document.createElement("tr");
      [String(row.step), row.date, fmt(row.point), fmt(row.lower), fmt(row.upper)].forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  function renderDiagnostics(report) {
    const series = report.series || {};
    const backtest = report.backtest || {};
    $("diag-freq").textContent = series.frequency || "—";
    $("diag-season").textContent = series.seasonal_period || "—";
    $("diag-folds").textContent = backtest.folds || "—";
    $("diag-mintrain").textContent = backtest.min_train || "—";

    const names = (report.ranking || []).map((r) => r.model).join(", ");
    $("diag-models").textContent = names ? `Methods compared: ${names}.` : "";
  }

  function fmt(value) {
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  // -----------------------------------------------------------------------
  // Chart
  // -----------------------------------------------------------------------
  function cssVar(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function renderChart(report) {
    const container = $("chart-container");
    if (typeof echarts === "undefined") {
      container.textContent = "Chart unavailable — the charting library failed to load.";
      return;
    }
    if (!chartInstance) chartInstance = echarts.init(container);

    const ink = cssVar("--ink", "#151C24");
    const graphite = cssVar("--graphite", "#59636E");
    const signal = cssVar("--signal", "#7C2D3A");
    const border = cssVar("--border", "#E0E4E7");

    const history = (report.series && report.series.history) || [];
    const historyDates = (report.series && report.series.history_dates) || [];
    const forecast = report.forecast || [];

    const categories = historyDates.concat(forecast.map((f) => f.date));
    const nHistory = history.length;

    const pad = (arr) => new Array(nHistory - 1).fill(null).concat(arr);
    // Repeat the last observed value so the forecast line joins the history.
    const forecastLine = pad([history[nHistory - 1]].concat(forecast.map((f) => f.point)));
    const lowerLine = pad([history[nHistory - 1]].concat(forecast.map((f) => f.lower)));
    const bandLine = pad(
      [0].concat(forecast.map((f) => f.upper - f.lower))
    );

    chartInstance.setOption(
      {
        animation: false,
        grid: { left: 62, right: 24, top: 24, bottom: 48 },
        tooltip: {
          trigger: "axis",
          backgroundColor: "#fff",
          borderColor: border,
          borderWidth: 1,
          textStyle: { color: ink, fontSize: 12 },
        },
        xAxis: {
          type: "category",
          data: categories,
          boundaryGap: false,
          axisLine: { lineStyle: { color: border } },
          axisLabel: { color: graphite, fontFamily: "IBM Plex Mono, monospace", fontSize: 11 },
        },
        yAxis: {
          type: "value",
          scale: true,
          splitLine: { lineStyle: { color: border, type: "dashed" } },
          axisLabel: {
            color: graphite,
            fontFamily: "IBM Plex Mono, monospace",
            fontSize: 11,
            formatter: (v) => Number(v).toLocaleString(),
          },
        },
        series: [
          {
            name: "History",
            type: "line",
            data: history,
            showSymbol: false,
            lineStyle: { color: graphite, width: 1.5 },
            // Vertical hairline at the point where history ends and the
            // forecast begins — the single most useful cue on this chart.
            markLine: {
              silent: true,
              symbol: "none",
              label: {
                formatter: "forecast starts",
                color: graphite,
                fontSize: 10,
                position: "insideEndTop",
              },
              lineStyle: { color: graphite, type: "dashed", width: 1, opacity: 0.7 },
              data: [{ xAxis: nHistory - 1 }],
            },
          },
          {
            name: "Range low",
            type: "line",
            data: lowerLine,
            stack: "band",
            showSymbol: false,
            lineStyle: { opacity: 0 },
            areaStyle: { opacity: 0 },
            silent: true,
            tooltip: { show: false },
          },
          {
            name: "95% range",
            type: "line",
            data: bandLine,
            stack: "band",
            showSymbol: false,
            lineStyle: { opacity: 0 },
            areaStyle: { color: signal, opacity: 0.12 },
            tooltip: { show: false },
          },
          {
            name: "Forecast",
            type: "line",
            data: forecastLine,
            showSymbol: true,
            symbolSize: 5,
            lineStyle: { color: signal, width: 2 },
            itemStyle: { color: signal },
          },
        ],
      },
      true
    );
  }

  // -----------------------------------------------------------------------
  // Exports
  // -----------------------------------------------------------------------
  async function exportReport(kind, filename) {
    if (!currentReport) return;
    try {
      const resp = await fetch(`/api/export/${kind}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentReport),
      });
      // Without this check a 500 downloads an HTML error page named .pdf.
      if (!resp.ok) throw new Error(`Export failed (${resp.status}).`);

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      showError(`Could not export: ${err.message}`);
      switchState("results");
    }
  }

  $("btn-export-csv").addEventListener("click", () => exportReport("csv", "forecast_report.csv"));
  $("btn-export-pdf").addEventListener("click", () => exportReport("pdf", "forecast_report.pdf"));

  // -----------------------------------------------------------------------
  // State machine
  // -----------------------------------------------------------------------
  function switchState(state) {
    ["input", "running", "download", "results"].forEach((name) => {
      const el = $(`state-${name}`);
      if (!el) return;
      el.classList.toggle("active", name === state);
      el.classList.toggle("hidden", name !== state);
    });
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    if (state === "results" && chartInstance) chartInstance.resize();
  }

  function showError(message) {
    const banner = $("input-error");
    banner.textContent = message;
    banner.classList.remove("hidden");
    banner.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  function clearError() {
    const banner = $("input-error");
    banner.textContent = "";
    banner.classList.add("hidden");
  }

  window.addEventListener("resize", () => {
    if (chartInstance) chartInstance.resize();
  });

  loadCapabilities();
});
