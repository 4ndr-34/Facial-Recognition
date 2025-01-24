document.getElementById("runButton").addEventListener("click", () => {
    const outputContent = document.getElementById("outputContent");

    // Clear previous output
    outputContent.textContent = "Running the notebook...";

    // Send a request to the Flask backend
    fetch("http://127.0.0.1:5000/face-recognition")
        .catch(error => {
            outputContent.textContent = `Error: ${error}`;
        });
});
