/**
 * Churn Prediction Dashboard — Application Logic
 * ================================================
 * Handles:
 *   - API communication (health, single predict, batch predict)
 *   - Feature engineering (raw customer → model features)
 *   - UI updates (gauge, factors, table, charts)
 */

const API_BASE = "";

// ─── Utility Helpers ──────────────────────────────────────────────────────────

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showToast(msg, type = "info") {
    const c = $("#toast-container");
    const t = document.createElement("div");
    t.className = `toast ${type}`;
    t.textContent = msg;
    c.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

// ─── Feature Engineering (mirrors src/preprocess.py + src/features.py) ────────
// This transforms human-friendly form data into the one-hot encoded +
// engineered feature dict that the API expects.

function engineerFeatures(raw) {
    const d = {};

    // --- Binary encodings ---
    d.gender = raw.gender === "Male" ? 1 : 0;
    d.SeniorCitizen = parseInt(raw.SeniorCitizen);
    d.Partner = raw.Partner === "Yes" ? 1 : 0;
    d.Dependents = raw.Dependents === "Yes" ? 1 : 0;
    d.PhoneService = raw.PhoneService === "Yes" ? 1 : 0;
    d.PaperlessBilling = raw.PaperlessBilling === "Yes" ? 1 : 0;

    // --- Numerics ---
    d.tenure = parseFloat(raw.tenure);
    d.MonthlyCharges = parseFloat(raw.MonthlyCharges);
    d.TotalCharges = parseFloat(raw.TotalCharges);

    // --- One-Hot: MultipleLines ---
    d["MultipleLines_No"] = raw.MultipleLines === "No" ? 1 : 0;
    d["MultipleLines_No phone service"] = raw.MultipleLines === "No phone service" ? 1 : 0;
    d["MultipleLines_Yes"] = raw.MultipleLines === "Yes" ? 1 : 0;

    // --- One-Hot: InternetService ---
    d["InternetService_DSL"] = raw.InternetService === "DSL" ? 1 : 0;
    d["InternetService_Fiber optic"] = raw.InternetService === "Fiber optic" ? 1 : 0;
    d["InternetService_No"] = raw.InternetService === "No" ? 1 : 0;

    // --- One-Hot: OnlineSecurity ---
    d["OnlineSecurity_No"] = raw.OnlineSecurity === "No" ? 1 : 0;
    d["OnlineSecurity_No internet service"] = raw.OnlineSecurity === "No internet service" ? 1 : 0;
    d["OnlineSecurity_Yes"] = raw.OnlineSecurity === "Yes" ? 1 : 0;

    // --- One-Hot: OnlineBackup ---
    d["OnlineBackup_No"] = raw.OnlineBackup === "No" ? 1 : 0;
    d["OnlineBackup_No internet service"] = raw.OnlineBackup === "No internet service" ? 1 : 0;
    d["OnlineBackup_Yes"] = raw.OnlineBackup === "Yes" ? 1 : 0;

    // --- One-Hot: DeviceProtection ---
    d["DeviceProtection_No"] = raw.DeviceProtection === "No" ? 1 : 0;
    d["DeviceProtection_No internet service"] = raw.DeviceProtection === "No internet service" ? 1 : 0;
    d["DeviceProtection_Yes"] = raw.DeviceProtection === "Yes" ? 1 : 0;

    // --- One-Hot: TechSupport ---
    d["TechSupport_No"] = raw.TechSupport === "No" ? 1 : 0;
    d["TechSupport_No internet service"] = raw.TechSupport === "No internet service" ? 1 : 0;
    d["TechSupport_Yes"] = raw.TechSupport === "Yes" ? 1 : 0;

    // --- One-Hot: StreamingTV ---
    d["StreamingTV_No"] = raw.StreamingTV === "No" ? 1 : 0;
    d["StreamingTV_No internet service"] = raw.StreamingTV === "No internet service" ? 1 : 0;
    d["StreamingTV_Yes"] = raw.StreamingTV === "Yes" ? 1 : 0;

    // --- One-Hot: StreamingMovies ---
    d["StreamingMovies_No"] = raw.StreamingMovies === "No" ? 1 : 0;
    d["StreamingMovies_No internet service"] = raw.StreamingMovies === "No internet service" ? 1 : 0;
    d["StreamingMovies_Yes"] = raw.StreamingMovies === "Yes" ? 1 : 0;

    // --- One-Hot: Contract ---
    d["Contract_Month-to-month"] = raw.Contract === "Month-to-month" ? 1 : 0;
    d["Contract_One year"] = raw.Contract === "One year" ? 1 : 0;
    d["Contract_Two year"] = raw.Contract === "Two year" ? 1 : 0;

    // --- One-Hot: PaymentMethod ---
    d["PaymentMethod_Bank transfer (automatic)"] = raw.PaymentMethod === "Bank transfer (automatic)" ? 1 : 0;
    d["PaymentMethod_Credit card (automatic)"] = raw.PaymentMethod === "Credit card (automatic)" ? 1 : 0;
    d["PaymentMethod_Electronic check"] = raw.PaymentMethod === "Electronic check" ? 1 : 0;
    d["PaymentMethod_Mailed check"] = raw.PaymentMethod === "Mailed check" ? 1 : 0;

    // ─── Engineered features (mirrors src/features.py) ───

    // tenure_group_num: 0–12→0, 13–24→1, 25–48→2, 49–60→3, 61+→4
    const t = d.tenure;
    if (t <= 12) d.tenure_group_num = 0;
    else if (t <= 24) d.tenure_group_num = 1;
    else if (t <= 48) d.tenure_group_num = 2;
    else if (t <= 60) d.tenure_group_num = 3;
    else d.tenure_group_num = 4;

    // service_count
    d.service_count = [
        d.PhoneService,
        d["MultipleLines_Yes"],
        d["OnlineSecurity_Yes"],
        d["OnlineBackup_Yes"],
        d["DeviceProtection_Yes"],
        d["TechSupport_Yes"],
        d["StreamingTV_Yes"],
        d["StreamingMovies_Yes"],
    ].reduce((s, v) => s + (v || 0), 0);

    // high_charges_no_support
    const hasSupport = (d["OnlineSecurity_Yes"] === 1) || (d["TechSupport_Yes"] === 1);
    d.high_charges_no_support = (d.MonthlyCharges > 65 && !hasSupport) ? 1 : 0;

    // charge_per_tenure
    d.charge_per_tenure = d.MonthlyCharges / (d.tenure + 1);

    // charges_ratio
    const expected = d.MonthlyCharges * d.tenure;
    d.charges_ratio = d.TotalCharges / (expected + 1);

    // non_auto_payment
    const isAuto = (raw.PaymentMethod === "Bank transfer (automatic)" ||
                    raw.PaymentMethod === "Credit card (automatic)");
    d.non_auto_payment = isAuto ? 0 : 1;

    // senior_alone
    d.senior_alone = (d.SeniorCitizen === 1 && d.Partner === 0) ? 1 : 0;

    return d;
}

// Prettify a feature name for display
function prettyFeatureName(name) {
    const map = {
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "Contract_Month-to-month": "Month-to-Month Contract",
        "Contract_One year": "One Year Contract",
        "Contract_Two year": "Two Year Contract",
        "PaymentMethod_Electronic check": "Electronic Check Payment",
        "PaymentMethod_Mailed check": "Mailed Check Payment",
        "PaymentMethod_Credit card (automatic)": "Credit Card (Auto)",
        "PaymentMethod_Bank transfer (automatic)": "Bank Transfer (Auto)",
        "InternetService_Fiber optic": "Fiber Optic Internet",
        "InternetService_DSL": "DSL Internet",
        "InternetService_No": "No Internet",
        "OnlineSecurity_Yes": "Online Security",
        "OnlineSecurity_No": "No Online Security",
        "TechSupport_Yes": "Tech Support",
        "TechSupport_No": "No Tech Support",
        "OnlineBackup_Yes": "Online Backup",
        "OnlineBackup_No": "No Online Backup",
        "DeviceProtection_Yes": "Device Protection",
        "DeviceProtection_No": "No Device Protection",
        "StreamingTV_Yes": "Streaming TV",
        "StreamingTV_No": "No Streaming TV",
        "StreamingMovies_Yes": "Streaming Movies",
        "StreamingMovies_No": "No Streaming Movies",
        "MultipleLines_Yes": "Multiple Lines",
        "MultipleLines_No": "Single Line",
        "tenure_group_num": "Tenure Group",
        "service_count": "Service Count",
        "high_charges_no_support": "High Charges (No Support)",
        "charge_per_tenure": "Charge / Tenure",
        "charges_ratio": "Charges Ratio",
        "non_auto_payment": "Non-Auto Payment",
        "senior_alone": "Senior Alone",
        "SeniorCitizen": "Senior Citizen",
        "PaperlessBilling": "Paperless Billing",
        "Partner": "Has Partner",
        "Dependents": "Has Dependents",
        "gender": "Gender",
        "PhoneService": "Phone Service",
    };
    return map[name] || name.replace(/_/g, " ");
}


// ─── API Calls ────────────────────────────────────────────────────────────────

async function checkHealth() {
    const badge = $("#api-status");
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.model_loaded) {
            badge.className = "status-badge";
            badge.querySelector(".status-text").textContent = "API Connected";
            const mb = $("#model-badge");
            mb.style.display = "inline-flex";
            $("#model-type").textContent = `${data.model_type} · ${data.n_features} features`;
        } else {
            badge.className = "status-badge status-checking";
            badge.querySelector(".status-text").textContent = "Model not loaded";
        }
    } catch {
        badge.className = "status-badge status-error";
        badge.querySelector(".status-text").textContent = "API Offline";
    }
}

async function predictSingle(features) {
    const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || `HTTP ${res.status}`);
    }
    return res.json();
}

async function predictBatchAPI(customers) {
    const res = await fetch(`${API_BASE}/predict/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(customers),
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || `HTTP ${res.status}`);
    }
    return res.json();
}


// ─── UI Updates ───────────────────────────────────────────────────────────────

function updateGauge(probability) {
    const pct = probability * 100;
    const totalLength = 251.33; // arc length in SVG
    const fillLength = totalLength * probability;
    const arc = $("#gauge-arc");
    arc.style.strokeDasharray = `${fillLength} ${totalLength}`;
    $("#gauge-value").textContent = `${pct.toFixed(1)}%`;
    $("#gauge-label").textContent = "Churn Probability";
}

function updateRiskBadge(riskLevel) {
    const badge = $("#risk-badge");
    badge.className = `risk-badge risk-${riskLevel.toLowerCase()}`;
    $("#risk-text").textContent = `${riskLevel} Risk`;
}

function updateFactors(factors) {
    const container = $("#factors-list");
    if (!factors || factors.length === 0) {
        container.innerHTML = `<p class="factors-empty">SHAP factors not available</p>`;
        return;
    }
    const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);
    container.innerHTML = factors.map(f => {
        const isPos = f.shap_value >= 0;
        const width = Math.min((Math.abs(f.shap_value) / maxAbs) * 100, 100);
        return `
            <div class="factor-item">
                <span class="factor-name" title="${f.feature}">${prettyFeatureName(f.feature)}</span>
                <div class="factor-bar-container">
                    <div class="factor-bar ${isPos ? 'positive' : 'negative'}" style="width: ${width}%"></div>
                </div>
                <span class="factor-value ${isPos ? 'positive' : 'negative'}">${isPos ? '+' : ''}${f.shap_value.toFixed(4)}</span>
            </div>`;
    }).join("");
}

function updateDetails(result) {
    const grid = $("#details-grid");
    const items = [
        { label: "Churn Probability", value: `${(result.churn_probability * 100).toFixed(2)}%` },
        { label: "Predicted Churn", value: result.predicted_churn === 1 ? "Yes ⚠️" : "No ✓" },
        { label: "Risk Level", value: result.risk_level },
        { label: "Latency", value: `${result.latency_ms} ms` },
    ];
    grid.innerHTML = items.map(i => `
        <div class="detail-item">
            <span class="detail-label">${i.label}</span>
            <span class="detail-value">${i.value}</span>
        </div>
    `).join("");
}

function showResults(result) {
    $("#results-placeholder").style.display = "none";
    const content = $("#results-content");
    content.style.display = "flex";

    updateGauge(result.churn_probability);
    updateRiskBadge(result.risk_level);
    updateFactors(result.top_risk_factors);
    updateDetails(result);
    $("#latency-value").textContent = `${result.latency_ms} ms`;
}


// ─── Batch UI ─────────────────────────────────────────────────────────────────

function showBatchResults(data) {
    const panel = $("#batch-results");
    panel.style.display = "block";

    // Summary
    $("#summary-total").textContent = data.n_customers;
    $("#summary-high").textContent = data.summary.high_risk;
    $("#summary-medium").textContent = data.summary.medium_risk;
    $("#summary-low").textContent = data.summary.low_risk;

    // Animate numbers
    animateCount("#summary-total", data.n_customers);
    animateCount("#summary-high", data.summary.high_risk);
    animateCount("#summary-medium", data.summary.medium_risk);
    animateCount("#summary-low", data.summary.low_risk);

    // Chart bars
    const total = data.n_customers || 1;
    const chartEl = $("#chart-bars");
    chartEl.innerHTML = [
        { label: "High Risk", count: data.summary.high_risk, cls: "high" },
        { label: "Medium Risk", count: data.summary.medium_risk, cls: "medium" },
        { label: "Low Risk", count: data.summary.low_risk, cls: "low" },
    ].map(r => `
        <div class="chart-row">
            <span class="chart-label">${r.label}</span>
            <div class="chart-track">
                <div class="chart-fill ${r.cls}" style="width: 0%">${r.count} (${((r.count / total) * 100).toFixed(1)}%)</div>
            </div>
        </div>
    `).join("");
    // Animate bars after paint
    requestAnimationFrame(() => {
        setTimeout(() => {
            chartEl.querySelectorAll(".chart-fill").forEach((bar, i) => {
                const counts = [data.summary.high_risk, data.summary.medium_risk, data.summary.low_risk];
                bar.style.width = `${Math.max((counts[i] / total) * 100, counts[i] > 0 ? 8 : 0)}%`;
            });
        }, 50);
    });

    $("#batch-latency").textContent = `Scored in ${data.latency_ms} ms`;

    // Table
    const tbody = $("#batch-tbody");
    tbody.innerHTML = data.predictions.map(p => `
        <tr>
            <td>${p.index + 1}</td>
            <td>${(p.churn_probability * 100).toFixed(2)}%</td>
            <td><span class="pill ${p.risk_level.toLowerCase()}"><span class="pill-dot"></span>${p.risk_level}</span></td>
            <td>${p.predicted_churn === 1 ? "Yes" : "No"}</td>
        </tr>
    `).join("");

    // Store for export
    window._batchData = data;
}

function animateCount(selector, target) {
    const el = $(selector);
    const duration = 600;
    const start = performance.now();
    const from = 0;
    function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        el.textContent = Math.round(from + (target - from) * ease);
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function exportCSV() {
    if (!window._batchData) return;
    const rows = [["Index", "Churn Probability", "Risk Level", "Predicted Churn"]];
    window._batchData.predictions.forEach(p => {
        rows.push([p.index, p.churn_probability, p.risk_level, p.predicted_churn]);
    });
    const csv = rows.map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "churn_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
    showToast("CSV exported successfully!", "success");
}


// ─── Sample batch data ───────────────────────────────────────────────────────

const SAMPLE_BATCH = [
    { gender: "Female", SeniorCitizen: "0", Partner: "No", Dependents: "No", tenure: "2", PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: "75.50", TotalCharges: "151.00" },
    { gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes", tenure: "48", PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "Yes", StreamingMovies: "No", Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)", MonthlyCharges: "85.00", TotalCharges: "4080.00" },
    { gender: "Female", SeniorCitizen: "1", Partner: "No", Dependents: "No", tenure: "5", PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: "95.00", TotalCharges: "475.00" },
    { gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "No", tenure: "36", PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "Yes", Contract: "One year", PaperlessBilling: "Yes", PaymentMethod: "Credit card (automatic)", MonthlyCharges: "70.00", TotalCharges: "2520.00" },
    { gender: "Female", SeniorCitizen: "0", Partner: "No", Dependents: "No", tenure: "1", PhoneService: "No", MultipleLines: "No phone service", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: "70.70", TotalCharges: "70.70" },
    { gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes", tenure: "60", PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)", MonthlyCharges: "95.50", TotalCharges: "5730.00" },
    { gender: "Female", SeniorCitizen: "1", Partner: "Yes", Dependents: "No", tenure: "18", PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "Yes", DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Mailed check", MonthlyCharges: "82.00", TotalCharges: "1476.00" },
    { gender: "Male", SeniorCitizen: "0", Partner: "No", Dependents: "No", tenure: "10", PhoneService: "Yes", MultipleLines: "No", InternetService: "No", OnlineSecurity: "No internet service", OnlineBackup: "No internet service", DeviceProtection: "No internet service", TechSupport: "No internet service", StreamingTV: "No internet service", StreamingMovies: "No internet service", Contract: "Month-to-month", PaperlessBilling: "No", PaymentMethod: "Mailed check", MonthlyCharges: "20.05", TotalCharges: "200.50" },
];


// ─── Event Handlers ───────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {

    // Health check
    checkHealth();
    setInterval(checkHealth, 30000);

    // Tabs
    $$(".tab").forEach(tab => {
        tab.addEventListener("click", () => {
            $$(".tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            $$(".panel").forEach(p => p.classList.remove("active"));
            $(`#panel-${tab.dataset.tab}`).classList.add("active");
        });
    });

    // Single prediction form
    $("#predict-form").addEventListener("submit", async e => {
        e.preventDefault();
        const btn = $("#btn-predict");
        const origHTML = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<div class="spinner"></div> Predicting…`;

        try {
            const formData = new FormData(e.target);
            const raw = Object.fromEntries(formData.entries());
            const features = engineerFeatures(raw);
            const result = await predictSingle(features);
            showResults(result);
            showToast("Prediction complete!", "success");
        } catch (err) {
            showToast(`Prediction failed: ${err.message}`, "error");
        } finally {
            btn.disabled = false;
            btn.innerHTML = origHTML;
        }
    });

    // File upload
    const uploadZone = $("#upload-zone");
    const fileInput = $("#file-input");

    uploadZone.addEventListener("click", () => fileInput.click());
    uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("dragover"); });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
    uploadZone.addEventListener("drop", e => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
    fileInput.addEventListener("change", e => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.name.endsWith(".json")) {
            showToast("Please upload a JSON file", "error");
            return;
        }
        const reader = new FileReader();
        reader.onload = e => {
            $("#batch-json").value = e.target.result;
            showToast(`Loaded ${file.name}`, "info");
        };
        reader.readAsText(file);
    }

    // Sample data
    $("#btn-load-sample").addEventListener("click", () => {
        $("#batch-json").value = JSON.stringify(SAMPLE_BATCH, null, 2);
        showToast("Sample data loaded — 8 customers", "info");
    });

    // Batch predict
    $("#btn-batch-predict").addEventListener("click", async () => {
        const btn = $("#btn-batch-predict");
        const origHTML = btn.innerHTML;

        let rawData;
        try {
            rawData = JSON.parse($("#batch-json").value);
        } catch {
            showToast("Invalid JSON. Please check your input.", "error");
            return;
        }

        if (!Array.isArray(rawData) || rawData.length === 0) {
            showToast("Input must be a non-empty JSON array", "error");
            return;
        }

        btn.disabled = true;
        btn.innerHTML = `<div class="spinner"></div> Scoring ${rawData.length} customers…`;

        try {
            // Engineer features for each customer
            const engineered = rawData.map(r => engineerFeatures(r));
            const result = await predictBatchAPI(engineered);
            showBatchResults(result);
            showToast(`Scored ${result.n_customers} customers!`, "success");
        } catch (err) {
            showToast(`Batch scoring failed: ${err.message}`, "error");
        } finally {
            btn.disabled = false;
            btn.innerHTML = origHTML;
        }
    });

    // Export CSV
    $("#btn-export").addEventListener("click", exportCSV);
});
