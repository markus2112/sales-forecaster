const DEFAULT_SERIES = [
    980000,
    1030000,
    1480000,
    1160000,
    1490000,
    1210000,
    1510000,
    1230000,
    1410000,
    1250000,
    1800000,
    1320000,
];

const DEFAULT_LABELS = [
    "May 1",
    "May 4",
    "May 6",
    "May 8",
    "May 10",
    "May 12",
    "May 14",
    "May 16",
    "May 19",
    "May 23",
    "May 27",
    "May 31",
];

const SCENARIO_TEMPLATES = [
    {
        title: "Competitor Price War",
        description: "Price drops force a sharper-than-expected revenue response",
        factor: -0.138,
        tag: "High Impact",
        tone: "high",
        probability: 0.32,
    },
    {
        title: "Supply Disruption (30%)",
        description: "Stock pressure weakens the revenue curve despite stable demand",
        factor: -0.117,
        tag: "Medium",
        tone: "medium",
        probability: 0.26,
    },
    {
        title: "Demand Spike (+25%)",
        description: "Unexpected demand surge lifts short-term sales performance",
        factor: 0.104,
        tag: "Medium",
        tone: "low",
        probability: 0.18,
    },
    {
        title: "Economic Downturn",
        description: "Market contraction increases downside exposure across the forecast",
        factor: -0.166,
        tag: "High Impact",
        tone: "high",
        probability: 0.17,
    },
    {
        title: "Marketing Boost (+20%)",
        description: "Incremental growth from stronger campaign conversion",
        factor: 0.071,
        tag: "Low",
        tone: "low",
        probability: 0.18,
    },
];

const state = {
    features: [],
    forecast: [],
    evaluations: null,
    models: {
        xgboost: false,
        lstm: false,
    },
    latestAnomaly: null,
};

const refs = {
    greetingTitle: document.getElementById("greetingTitle"),
    apiStatusPill: document.getElementById("apiStatusPill"),
    lastActionPill: document.getElementById("lastActionPill"),
    chartSourceNote: document.getElementById("chartSourceNote"),
    totalForecastValue: document.getElementById("totalForecastValue"),
    totalForecastTrend: document.getElementById("totalForecastTrend"),
    accuracyValue: document.getElementById("accuracyValue"),
    accuracyTrend: document.getElementById("accuracyTrend"),
    robustnessValue: document.getElementById("robustnessValue"),
    robustnessTrend: document.getElementById("robustnessTrend"),
    worstCaseValue: document.getElementById("worstCaseValue"),
    worstCaseTrend: document.getElementById("worstCaseTrend"),
    activeScenariosValue: document.getElementById("activeScenariosValue"),
    activeScenariosTrend: document.getElementById("activeScenariosTrend"),
    trendChart: document.getElementById("trendChart"),
    insightText: document.getElementById("insightText"),
    deviationValue: document.getElementById("deviationValue"),
    confidenceValue: document.getElementById("confidenceValue"),
    robustnessRing: document.getElementById("robustnessRing"),
    robustnessScoreLarge: document.getElementById("robustnessScoreLarge"),
    robustnessLabel: document.getElementById("robustnessLabel"),
    robustnessDescription: document.getElementById("robustnessDescription"),
    stressList: document.getElementById("stressList"),
    scenarioList: document.getElementById("scenarioList"),
    featureCount: document.getElementById("featureCount"),
    featureCountDisplay: document.getElementById("featureCountDisplay"),
    modelState: document.getElementById("modelState"),
    modelStateDisplay: document.getElementById("modelStateDisplay"),
    anomalyState: document.getElementById("anomalyState"),
    anomalyStateDisplay: document.getElementById("anomalyStateDisplay"),
    activityList: document.getElementById("activityList"),
    actionButtons: document.querySelectorAll("[data-action][data-endpoint]"),
    scrollButtons: document.querySelectorAll("[data-scroll-target]"),
};

function setGreeting() {
    const hour = new Date().getHours();
    let greeting = "Hello";

    if (hour < 12) {
        greeting = "Good morning";
    } else if (hour < 18) {
        greeting = "Good afternoon";
    } else {
        greeting = "Good evening";
    }

    refs.greetingTitle.textContent = `${greeting}, Ronak!`;
}

function setStatusPill(element, text, tone = "muted") {
    element.textContent = text;
    element.className = `status-pill ${tone}`;
}

function updateLastAction(message) {
    const time = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
    });
    refs.lastActionPill.textContent = `${message} at ${time}`;
}

function pushActivity(message) {
    const time = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
    });

    if (
        refs.activityList.children.length === 1 &&
        refs.activityList.children[0].textContent === "Waiting for the first action."
    ) {
        refs.activityList.innerHTML = "";
    }

    const item = document.createElement("li");
    const stamp = document.createElement("strong");
    stamp.textContent = time;
    item.appendChild(stamp);
    item.appendChild(document.createTextNode(` ${message}`));
    refs.activityList.prepend(item);

    while (refs.activityList.children.length > 6) {
        refs.activityList.removeChild(refs.activityList.lastChild);
    }
}

function setButtonBusy(button, isBusy) {
    button.classList.toggle("is-loading", isBusy);
    button.disabled = isBusy;
}

function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "N/A";
    }

    return `${Number(value).toFixed(2)}%`;
}

function formatCompactCurrency(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "N/A";
    }

    return new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: "USD",
        notation: "compact",
        maximumFractionDigits: 2,
    }).format(Number(value));
}

async function apiRequest(endpoint, options = {}) {
    const response = await fetch(endpoint, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
        ...options,
    });

    const payload = await response.json();

    if (!response.ok) {
        throw new Error(payload.detail || payload.error || "Request failed");
    }

    if (payload.error) {
        throw new Error(payload.error);
    }

    return payload;
}

function formatDateLabel(value) {
    if (!value) {
        return "N/A";
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
    });
}

function getChartDataset() {
    if (state.features.length >= 6) {
        const rows = state.features.slice(-12);
        return {
            source: "live",
            labels: rows.map((row) => formatDateLabel(row.date)),
            actual: rows.map((row) => Number(row.sales)),
        };
    }

    return {
        source: "preview",
        labels: DEFAULT_LABELS,
        actual: DEFAULT_SERIES,
    };
}

function getForecastTail(actualSeries) {
    if (state.forecast.length) {
        return state.forecast.map((value) => Number(value));
    }

    const seed = actualSeries.slice(-5);
    return seed.map((value, index) => Math.round(value * (1.02 + index * 0.017)));
}

function getForecastComposition(actualSeries) {
    const tail = getForecastTail(actualSeries);
    const tailLength = Math.min(tail.length, Math.max(4, Math.min(5, actualSeries.length)));
    const head = actualSeries.slice(0, actualSeries.length - tailLength);

    return {
        series: [...head, ...tail.slice(-tailLength)],
        tailStart: head.length,
    };
}

function getStressFactor() {
    if (state.evaluations && state.evaluations.Hybrid && state.evaluations.Hybrid.MAPE !== null) {
        return Math.min(0.22, Math.max(0.08, Number(state.evaluations.Hybrid.MAPE) / 100 + 0.04));
    }

    return 0.138;
}

function getWorstCaseSeries(forecastSeries, tailStart) {
    const factor = getStressFactor();
    return forecastSeries.map((value, index) => {
        if (index < tailStart) {
            return null;
        }

        const slopePenalty = (index - tailStart) * 0.012;
        return Math.round(value * (1 - factor - slopePenalty));
    });
}

function createPointString(series, scaleX, scaleY) {
    return series
        .map((value, index) => {
            if (value === null || value === undefined) {
                return null;
            }

            return `${scaleX(index)},${scaleY(value)}`;
        })
        .filter(Boolean)
        .join(" ");
}

function calculateRobustnessScore() {
    if (state.evaluations && state.evaluations.Hybrid && state.evaluations.Hybrid.MAPE !== null) {
        return Math.max(52, Math.min(96, Math.round(100 - Number(state.evaluations.Hybrid.MAPE) * 1.5)));
    }

    if (state.models.xgboost && state.models.lstm) {
        return 84;
    }

    if (state.models.xgboost || state.models.lstm) {
        return 72;
    }

    return 87;
}

function getScenarioData() {
    const forecastSeries = getForecastComposition(getChartDataset().actual).series;
    const totalBase = forecastSeries.slice(-5).reduce((sum, value) => sum + value, 0);

    return SCENARIO_TEMPLATES.map((scenario) => ({
        ...scenario,
        delta: Math.round(totalBase * scenario.factor),
    }));
}

function renderTrendChart() {
    const dataset = getChartDataset();
    const actualSeries = dataset.actual;
    const forecastInfo = getForecastComposition(actualSeries);
    const worstSeries = getWorstCaseSeries(forecastInfo.series, forecastInfo.tailStart);

    const allValues = [...actualSeries, ...forecastInfo.series, ...worstSeries.filter((value) => value !== null)];
    const maxValue = Math.max(...allValues);
    const minValue = Math.min(...allValues);
    const yMin = Math.max(0, minValue * 0.75);
    const yMax = maxValue * 1.12;
    const chartWidth = 760;
    const chartHeight = 320;
    const padding = { top: 24, right: 22, bottom: 42, left: 68 };

    const scaleX = (index) => {
        const range = chartWidth - padding.left - padding.right;
        return padding.left + (index / (actualSeries.length - 1)) * range;
    };

    const scaleY = (value) => {
        const range = chartHeight - padding.top - padding.bottom;
        const normalized = (value - yMin) / (yMax - yMin || 1);
        return chartHeight - padding.bottom - normalized * range;
    };

    const actualPoints = createPointString(actualSeries, scaleX, scaleY);
    const forecastPoints = createPointString(forecastInfo.series, scaleX, scaleY);
    const worstPoints = createPointString(worstSeries, scaleX, scaleY);
    const gridLines = [];
    const xLabels = [];

    for (let tick = 0; tick <= 4; tick += 1) {
        const value = yMin + ((yMax - yMin) / 4) * tick;
        const y = scaleY(value);
        gridLines.push(`
            <line x1="${padding.left}" y1="${y}" x2="${chartWidth - padding.right}" y2="${y}" stroke="rgba(109, 78, 255, 0.08)" stroke-width="1"></line>
            <text x="${padding.left - 12}" y="${y + 4}" text-anchor="end" fill="#6d7593" font-size="12">${formatCompactCurrency(value)}</text>
        `);
    }

    dataset.labels.forEach((label, index) => {
        if (index % 2 === 0 || index === dataset.labels.length - 1) {
            xLabels.push(`
                <text x="${scaleX(index)}" y="${chartHeight - 14}" text-anchor="middle" fill="#6d7593" font-size="12">${label}</text>
            `);
        }
    });

    refs.trendChart.innerHTML = `
        <defs>
            <linearGradient id="actualGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#6d4eff"></stop>
                <stop offset="100%" stop-color="#8a71ff"></stop>
            </linearGradient>
            <linearGradient id="forecastGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#4b67ff"></stop>
                <stop offset="100%" stop-color="#6ea3ff"></stop>
            </linearGradient>
            <linearGradient id="worstGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#ff818d"></stop>
                <stop offset="100%" stop-color="#ff5f6c"></stop>
            </linearGradient>
        </defs>
        ${gridLines.join("")}
        <polyline fill="none" stroke="url(#actualGradient)" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" points="${actualPoints}"></polyline>
        <polyline fill="none" stroke="url(#forecastGradient)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="7 7" points="${forecastPoints}"></polyline>
        <polyline fill="none" stroke="url(#worstGradient)" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="6 6" points="${worstPoints}"></polyline>
        ${forecastInfo.series
            .map((value, index) => {
                if (index < forecastInfo.tailStart) {
                    return "";
                }

                return `<circle cx="${scaleX(index)}" cy="${scaleY(value)}" r="3.2" fill="#4b67ff"></circle>`;
            })
            .join("")}
        ${xLabels.join("")}
    `;

    refs.chartSourceNote.textContent =
        dataset.source === "live"
            ? `Showing the latest ${actualSeries.length} live points from your feature pipeline.`
            : "Rendering a polished preview until live data is loaded.";

    const delta = forecastInfo.series[forecastInfo.series.length - 1] - actualSeries[actualSeries.length - 1];
    const deltaPct = Math.abs(delta / (actualSeries[actualSeries.length - 1] || 1)) * 100;
    refs.deviationValue.textContent = `${dataset.labels[dataset.labels.length - 1]} | ${delta >= 0 ? "+" : ""}${deltaPct.toFixed(1)}%`;

    const averageActual = actualSeries.reduce((sum, value) => sum + value, 0) / actualSeries.length;
    const averageForecast = forecastInfo.series.reduce((sum, value) => sum + value, 0) / forecastInfo.series.length;
    refs.insightText.textContent =
        averageForecast >= averageActual
            ? "Forecast tracking slightly above actual demand."
            : "Forecast trailing recent demand and may need recalibration.";

    const confidence = Math.max(62, Math.min(96, Math.round(calculateRobustnessScore() - getStressFactor() * 40)));
    refs.confidenceValue.textContent = `${confidence}%`;
}

function renderScenarioList() {
    const scenarios = getScenarioData();
    refs.scenarioList.innerHTML = scenarios
        .map((scenario) => `
            <article class="scenario-item">
                <div class="scenario-icon"></div>
                <div class="scenario-copy">
                    <strong>${scenario.title}</strong>
                    <p>${scenario.description}</p>
                    <span class="tag ${scenario.tone}">${scenario.tag}</span>
                </div>
                <div class="scenario-meta">
                    <strong>${scenario.delta >= 0 ? "+" : ""}${formatCompactCurrency(scenario.delta)}</strong>
                    <span>${Math.round(scenario.probability * 100)}% prob.</span>
                </div>
            </article>
        `)
        .join("");
}

function renderRobustnessPanel() {
    const score = calculateRobustnessScore();
    refs.robustnessRing.style.setProperty("--score", score);
    refs.robustnessScoreLarge.textContent = score;

    let label = "High Robustness";
    let description = "Model performs well under adversarial conditions and maintains stable output quality.";

    if (score < 70) {
        label = "Needs Tuning";
        description = "Stress sensitivity is elevated and the current setup should be reviewed.";
    } else if (score < 84) {
        label = "Stable Under Stress";
        description = "The forecasting pipeline remains usable, with a few downside cases to monitor closely.";
    }

    refs.robustnessLabel.textContent = label;
    refs.robustnessDescription.textContent = description;

    const downsideScenarios = getScenarioData()
        .filter((scenario) => scenario.factor < 0)
        .slice(0, 4);

    refs.stressList.innerHTML = downsideScenarios
        .map((scenario) => {
            const percent = Math.abs(scenario.factor * 100);
            const width = Math.min(100, percent * 4.3);
            return `
                <div class="stress-item">
                    <div class="stress-meta">
                        <span>${scenario.title}</span>
                        <strong class="negative">-${percent.toFixed(1)}%</strong>
                    </div>
                    <div class="stress-bar">
                        <span style="width:${width}%"></span>
                    </div>
                </div>
            `;
        })
        .join("");
}

function updateStatusPanel() {
    const featureCount = String(state.features.length);
    refs.featureCount.textContent = featureCount;
    refs.featureCountDisplay.textContent = featureCount;

    let modelState = "Waiting";
    if (state.models.xgboost && state.models.lstm) {
        modelState = "Hybrid Ready";
    } else if (state.models.xgboost || state.models.lstm) {
        modelState = "Partially Trained";
    }
    refs.modelState.textContent = modelState;
    refs.modelStateDisplay.textContent = modelState;

    const anomalyState = state.latestAnomaly === null ? "None" : state.latestAnomaly ? "Detected" : "Clear";
    refs.anomalyState.textContent = anomalyState;
    refs.anomalyStateDisplay.textContent = anomalyState;
}

function setTrendTone(element, tone, text) {
    element.textContent = text;
    element.classList.remove("positive", "negative", "neutral");
    element.classList.add(tone);
}

function renderKpis() {
    const forecastTail = getForecastTail(getChartDataset().actual);
    const forecastTotal = forecastTail.reduce((sum, value) => sum + value, 0);
    const recentActual = getChartDataset().actual.slice(-5).reduce((sum, value) => sum + value, 0);
    const accuracy = state.evaluations && state.evaluations.Hybrid ? state.evaluations.Hybrid.MAPE : 8.74;
    const robustness = calculateRobustnessScore();
    const scenarios = getScenarioData();
    const downside = scenarios.filter((scenario) => scenario.delta < 0);
    const worstCase = downside.length ? Math.min(...downside.map((scenario) => scenario.delta)) : 0;
    const trend = ((forecastTotal - recentActual) / (recentActual || 1)) * 100;

    refs.totalForecastValue.textContent = formatCompactCurrency(forecastTotal);
    refs.accuracyValue.textContent = formatPercent(accuracy);
    refs.robustnessValue.textContent = `${robustness} / 100`;
    refs.worstCaseValue.textContent = `${worstCase >= 0 ? "+" : ""}${formatCompactCurrency(worstCase)}`;
    refs.activeScenariosValue.textContent = String(scenarios.length);

    setTrendTone(
        refs.totalForecastTrend,
        trend >= 0 ? "positive" : "negative",
        `${trend >= 0 ? "+" : ""}${trend.toFixed(1)}% vs baseline`
    );
    setTrendTone(
        refs.accuracyTrend,
        accuracy <= 9 ? "positive" : "negative",
        accuracy <= 9 ? "Healthy hybrid error" : "Forecast drift detected"
    );
    setTrendTone(
        refs.robustnessTrend,
        robustness >= 84 ? "positive" : robustness >= 70 ? "neutral" : "negative",
        robustness >= 84 ? "Stable under stress" : robustness >= 70 ? "Monitoring watchlist" : "Needs attention"
    );
    setTrendTone(refs.worstCaseTrend, "negative", "Downside exposure monitored");
    setTrendTone(refs.activeScenariosTrend, "neutral", `${scenarios.length} scenarios active`);
}

function renderDashboard() {
    renderKpis();
    renderTrendChart();
    renderRobustnessPanel();
    renderScenarioList();
    updateStatusPanel();
}

async function loadHealth() {
    try {
        const payload = await apiRequest("/api/health");
        setStatusPill(refs.apiStatusPill, payload.message, "success");
        pushActivity("API health check completed successfully.");
    } catch (error) {
        setStatusPill(refs.apiStatusPill, "API unavailable", "error");
        pushActivity(`API health check failed: ${error.message}`);
    }
}

async function loadModelStatus() {
    try {
        const payload = await apiRequest("/api/status");
        state.models.xgboost = payload.xgboost_loaded;
        state.models.lstm = payload.lstm_loaded;
        renderDashboard();

        if (payload.xgboost_loaded || payload.lstm_loaded) {
            pushActivity("Detected saved model state from backend startup.");
        }
    } catch (error) {
        pushActivity(`Model status check failed: ${error.message}`);
    }
}

async function loadFeatures(options = {}) {
    try {
        const payload = await apiRequest("/generate-features/");
        state.features = Array.isArray(payload) ? payload : [];
        renderDashboard();

        if (!options.silent) {
            pushActivity(`Loaded ${state.features.length} feature rows from the backend.`);
        }
    } catch (error) {
        if (!options.silent) {
            pushActivity(`Feature preview unavailable: ${error.message}`);
        }
    }
}

async function handleWorkflowAction(button) {
    const endpoint = button.dataset.endpoint;
    const action = button.dataset.action;
    const label = button.textContent.trim().replace(/\s+/g, " ");

    setButtonBusy(button, true);

    try {
        const payload = await apiRequest(endpoint);

        if (action === "features") {
            state.features = Array.isArray(payload) ? payload : [];
            pushActivity(`Feature generation completed with ${state.features.length} rows returned.`);
        }

        if (action === "xgboost") {
            state.models.xgboost = true;
            pushActivity(payload.message || "XGBoost training finished.");
        }

        if (action === "lstm") {
            state.models.lstm = true;
            pushActivity(payload.message || "LSTM training finished.");
        }

        if (action === "forecast") {
            state.forecast = Array.isArray(payload.hybrid_prediction) ? payload.hybrid_prediction : [];
            pushActivity(`Hybrid forecast completed with ${payload.count || state.forecast.length} predictions.`);
        }

        if (action === "evaluate") {
            state.evaluations = payload;
            pushActivity("Model evaluation completed.");
        }

        updateLastAction(label);
        renderDashboard();
    } catch (error) {
        updateLastAction(`${label} failed`);
        pushActivity(`${label} failed: ${error.message}`);
    } finally {
        setButtonBusy(button, false);
    }
}

function scrollToTarget(targetId) {
    const target = document.getElementById(targetId);
    if (!target) {
        return;
    }

    target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function bindScrollButtons() {
    refs.scrollButtons.forEach((button) => {
        button.addEventListener("click", () => {
            scrollToTarget(button.dataset.scrollTarget);
        });
    });
}

function bindActionButtons() {
    refs.actionButtons.forEach((button) => {
        button.addEventListener("click", () => handleWorkflowAction(button));
    });
}

function highlightNavOnScroll() {
    const sections = [
        "heroBanner",
        "kpiSection",
        "forecastPanel",
        "scenarioPanel",
        "quickActions",
        "robustnessPanel",
    ]
        .map((id) => document.getElementById(id))
        .filter(Boolean);
    const navButtons = Array.from(document.querySelectorAll(".nav-link"));

    window.addEventListener("scroll", () => {
        const position = window.scrollY + 140;
        let activeId = "heroBanner";

        sections.forEach((section) => {
            if (section.offsetTop <= position) {
                activeId = section.id;
            }
        });

        navButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.scrollTarget === activeId);
        });
    });
}

function init() {
    setGreeting();
    renderDashboard();
    loadHealth();
    loadModelStatus();
    loadFeatures({ silent: true });
    bindActionButtons();
    bindScrollButtons();
    highlightNavOnScroll();
    pushActivity("Dashboard initialized in preview mode.");
}

init();
