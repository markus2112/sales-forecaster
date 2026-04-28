document.addEventListener("DOMContentLoaded", function () {

    // ================= GLOBAL STATE =================
    let latestForecast = [];
    let latestFeatures = [];
    let forecastChart = null;
    let historicalChart = null;

    // ================= INITIALIZE CHARTS =================
    function initCharts() {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Outfit';

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 10
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(11, 9, 20, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#c4b5fd',
                    borderColor: 'rgba(139, 92, 246, 0.2)',
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 8
                }
            },
            scales: {
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    }
                },
                x: {
                    grid: {
                        display: false,
                        drawBorder: false
                    }
                }
            }
        };

        const ctxForecast = document.getElementById('forecastVsActualChart').getContext('2d');
        forecastChart = new Chart(ctxForecast, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Forecasted (Hybrid)',
                        data: [],
                        borderColor: '#8b5cf6',
                        borderWidth: 2,
                        tension: 0.4,
                        pointBackgroundColor: '#8b5cf6',
                        fill: true,
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    }
                ]
            },
            options: chartOptions
        });

        const ctxHistorical = document.getElementById('historicalChart').getContext('2d');
        historicalChart = new Chart(ctxHistorical, {
            type: 'bar',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Historical Sales',
                    data: [1200, 1500, 1800, 1400, 2500, 2100], // Mock data until fetched
                    backgroundColor: 'rgba(139, 92, 246, 0.7)',
                    borderRadius: 4,
                    barPercentage: 0.5
                }]
            },
            options: chartOptions
        });
    }

    initCharts();

    // ================= API HEALTH =================
    async function checkAPI() {
        try {
            const response = await fetch("/api/health");
            const data = await response.json();

            document.getElementById("apiStatusPill").innerHTML =
                (data.message || "API Connected") + ' <span class="dot"></span>';

        } catch (error) {
            document.getElementById("apiStatusPill").innerHTML =
                "API Not Connected";
            document.getElementById("apiStatusPill").classList.remove("online");
            document.getElementById("apiStatusPill").style.color = "var(--danger)";
        }
    }

    // ================= SIMPLE MESSAGE (TOAST) =================
    function showMessage(message) {
        const tracker = document.getElementById("lastActionPill");
        tracker.innerText = message;
        
        // Highlight action
        tracker.style.color = '#fff';
        tracker.style.textShadow = '0 0 10px #8b5cf6';
        setTimeout(() => {
            tracker.style.color = 'var(--accent)';
            tracker.style.textShadow = 'none';
        }, 1000);

        // Show Toast
        const toast = document.getElementById("toast");
        document.getElementById("toastMessage").innerText = message;
        toast.classList.add("show");
        
        setTimeout(() => {
            toast.classList.remove("show");
        }, 3000);
    }

    // ================= ADD SALES FORM =================
    const salesForm = document.getElementById("salesForm");

    if (salesForm) {
        salesForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            const payload = {
                date: document.getElementById("date").value,
                sales: parseFloat(document.getElementById("sales").value),
                stock: parseInt(document.getElementById("stock").value),
                promotion: document.getElementById("promotion").checked,
                holiday: document.getElementById("holiday").checked
            };

            showMessage("Submitting sales data...");

            try {
                const response = await fetch("/add-sales/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();

                if (result.error) {
                    showMessage("Error: " + result.error);
                    return;
                }

                showMessage("Sales data inserted successfully");

                if (result.anomaly === true) {
                    showMessage("Warning: Anomaly detected in sales data");
                }

                salesForm.reset();

            } catch (error) {
                console.log(error);
                showMessage("Server error while inserting sales");
            }
        });
    }

    // ================= EXCEL / CSV UPLOAD =================
    const uploadBtn = document.getElementById("uploadExcelBtn");

    if (uploadBtn) {
        uploadBtn.addEventListener("click", async function () {

            const fileInput = document.getElementById("excelFile");
            const status = document.getElementById("uploadStatus");

            if (!fileInput.files.length) {
                status.innerText = "Please select a file first";
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            status.innerText = "Uploading...";
            showMessage("Uploading Excel file...");

            try {
                const response = await fetch("/upload-excel/", {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();

                if (result.error) {
                    status.innerText = "Error: " + result.error;
                    showMessage("Error: " + result.error);
                    return;
                }

                status.innerText = result.message;
                showMessage(result.message);

                fileInput.value = "";

            } catch (error) {
                console.log(error);
                status.innerText = "Upload failed";
                showMessage("Excel upload failed");
            }
        });
    }

    // ================= GENERATE FEATURES =================
    async function generateFeatures() {
        showMessage("Generating features...");
        try {
            const response = await fetch("/generate-features/");
            const data = await response.json();

            if (data.error) {
                showMessage(data.error);
                return;
            }

            latestFeatures = data;
            showMessage("Features generated successfully");
            console.log("Generated Features:", data);

        } catch (error) {
            console.log(error);
            showMessage("Feature generation failed");
        }
    }

    // ================= TRAIN XGBOOST =================
    async function trainXGBoost() {
        showMessage("Training XGBoost model...");
        try {
            const response = await fetch("/train-xgboost/");
            const data = await response.json();

            if (data.error) {
                showMessage(data.error);
                return;
            }

            showMessage(data.message || "XGBoost trained successfully");

        } catch (error) {
            console.log(error);
            showMessage("XGBoost training failed");
        }
    }

    // ================= TRAIN LSTM =================
    async function trainLSTM() {
        showMessage("Training LSTM model...");
        try {
            const response = await fetch("/train-lstm/");
            const data = await response.json();

            if (data.error) {
                showMessage(data.error);
                return;
            }

            showMessage(data.message || "LSTM trained successfully");

        } catch (error) {
            console.log(error);
            showMessage("LSTM training failed");
        }
    }

    // ================= HYBRID FORECAST =================
    async function hybridForecast() {
        showMessage("Generating Hybrid Forecast...");
        try {
            const response = await fetch("/hybrid-forecast/");
            const data = await response.json();

            if (data.error) {
                showMessage(data.error);
                return;
            }

            latestForecast = data.hybrid_prediction || [];

            if (latestForecast.length) {
                const total = latestForecast.reduce((a, b) => a + b, 0);
                document.getElementById("totalForecastValue").innerText =
                    "$" + total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                
                // Update chart
                forecastChart.data.labels = latestForecast.map((_, i) => `T+${i+1}`);
                forecastChart.data.datasets[0].data = latestForecast;
                forecastChart.update();

                // Update future forecast list
                const list = document.getElementById("futureForecastList");
                list.innerHTML = "";
                latestForecast.slice(0, 5).forEach((val, i) => {
                    const item = document.createElement("div");
                    item.className = "forecast-item";
                    item.innerHTML = `
                        <div class="date">Step ${i+1}</div>
                        <div class="proj-value">$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                    `;
                    list.appendChild(item);
                });
            }

            showMessage("Hybrid forecast generated successfully");
            console.log("Forecast:", data);

        } catch (error) {
            console.log(error);
            showMessage("Forecast generation failed");
        }
    }

    // ================= EVALUATE MODELS =================
    async function evaluateModels() {
        showMessage("Evaluating models...");
        try {
            const response = await fetch("/evaluate-models/");
            const data = await response.json();

            if (data.error) {
                showMessage(data.error);
                return;
            }

            if (
                data.Hybrid &&
                data.Hybrid.MAPE !== null &&
                data.Hybrid.MAPE !== undefined
            ) {
                document.getElementById("accuracyValue").innerText =
                    data.Hybrid.MAPE.toFixed(2) + "%";
            }

            showMessage("Model evaluation completed");
            console.log("Evaluation:", data);

        } catch (error) {
            console.log(error);
            showMessage("Model evaluation failed");
        }
    }

    // ================= BUTTON EVENTS =================
    const buttons = document.querySelectorAll("[data-action]");

    buttons.forEach(button => {
        button.addEventListener("click", function () {
            const action = button.dataset.action;

            if (action === "features") generateFeatures();
            if (action === "xgboost") trainXGBoost();
            if (action === "lstm") trainLSTM();
            if (action === "forecast") hybridForecast();
            if (action === "evaluate") evaluateModels();
        });
    });

    // ================= NAV SMOOTH SCROLL =================
    document.querySelectorAll('.nav-item').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if(href && href.startsWith('#')) {
                e.preventDefault();
                document.querySelector(href).scrollIntoView({
                    behavior: 'smooth'
                });
                
                // Update active state
                document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });

    // ================= START =================
    checkAPI();

});