document.getElementById("uploadForm").onsubmit = async function (event) {
    event.preventDefault();

    let fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select a file first.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Show loading animation
    document.getElementById("loading").style.display = "block";
    document.getElementById("result").style.display = "none";

    try {
        let response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        let contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
            let data = await response.json();
            document.getElementById("result").innerHTML = `
                <h4>Prediction:</h4>
                <p>${data.prediction}</p>
            `;
        } else {
            let text = await response.text();
            document.getElementById("result").innerHTML = `
                <h4>Response:</h4>
                <p>${text}</p>
            `;
        }
    } catch (err) {
        console.error("Fetch error:", err);
        document.getElementById("result").innerHTML = `<p>❌ Something went wrong.</p>`;
    }

    document.getElementById("loading").style.display = "none";
    document.getElementById("result").style.display = "block";
};
