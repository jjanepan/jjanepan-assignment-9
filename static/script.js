document.getElementById("experiment-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const activation = document.getElementById("activation").value.trim();
    const lr = parseFloat(document.getElementById("lr").value);
    const stepNum = parseInt(document.getElementById("step_num").value);

    // Validate input
    if (!validateInput(activation, lr, stepNum)) {
        return; // Stop execution if validation fails
    }

    // Show loading indicator
    toggleLoading(true);

    try {
        // Send experiment data to the server
        const response = await fetch("/run_experiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ activation, lr, step_num: stepNum }),
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();

        // Display results
        const resultsDiv = document.getElementById("results");
        const resultImg = document.getElementById("result_gif");

        if (data.result_gif) {
            resultsDiv.style.display = "block";
            resultImg.src = data.result_gif;
            resultImg.style.display = "block";
        } else {
            alert("No result image available. Please try again.");
        }
    } catch (error) {
        console.error("Error running experiment:", error);
        alert("An error occurred while running the experiment. Please try again later.");
    } finally {
        // Hide loading indicator
        toggleLoading(false);
    }
});

// Validation function
function validateInput(activation, lr, stepNum) {
    const validActivations = ["relu", "tanh", "sigmoid"];

    if (!validActivations.includes(activation)) {
        alert("Please choose a valid activation function: relu, tanh, or sigmoid.");
        return false;
    }

    if (isNaN(lr) || lr <= 0) {
        alert("Please enter a positive number for the learning rate.");
        return false;
    }

    if (isNaN(stepNum) || stepNum <= 0) {
        alert("Please enter a positive integer for the number of training steps.");
        return false;
    }

    return true;
}

// Toggle loading indicator
function toggleLoading(show) {
    const loadingDiv = document.getElementById("loading");
    loadingDiv.style.display = show ? "block" : "none";
}
