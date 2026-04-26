document.addEventListener("DOMContentLoaded", function () {

    // ================= GLOBAL STATE =================
    let latestForecast = [];
    let latestFeatures = [];

    // ================= API HEALTH =================
    async function checkAPI() {
        try {
            const response = await fetch("/api/health");
            const data = await response.json();

            document.getElementById("apiStatusPill").innerText =
                data.message || "API Connected";

        } catch (error) {
            document.getElementById("apiStatusPill").innerText =
                "API Not Connected";
        }
    }

    // ================= SIMPLE MESSAGE =================
    function showMessage(message) {
        document.getElementById("lastActionPill").innerText = message;
        alert(message);
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
                    alert("Warning: Anomaly detected in sales data");
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

            try {
                const response = await fetch("/upload-excel/", {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();

                if (result.error) {
                    status.innerText = "Error: " + result.error;
                    showMessage(result.error);
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
                    "$" + total.toFixed(2);
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

            if (action === "features") {
                generateFeatures();
            }

            if (action === "xgboost") {
                trainXGBoost();
            }

            if (action === "lstm") {
                trainLSTM();
            }

            if (action === "forecast") {
                hybridForecast();
            }

            if (action === "evaluate") {
                evaluateModels();
            }
        });
    });

    // ================= START =================
    checkAPI();

});