const form = document.getElementById('upload-form');
const filenameLabel = document.getElementById('file-label');
const resultDiv = document.getElementById('result');
const diseaseNameSpan = document.getElementById('disease-name');
const fertilizerSpan = document.getElementById('fertilizer');
const treatmentSpan = document.getElementById('treatment');
const imagePreview = document.getElementById('image-preview');
const filename = document.getElementById('file-name');
const spinner = document.getElementById('spinner');



// Show file name and preview when a file is selected
document.getElementById('file').onchange = (event) => {
    const fileName = event.target.files[0].name;

    // Display the file name below the image preview
    const fileNameDiv = document.getElementById('file-name');
    fileNameDiv.innerText = fileName;
    fileNameDiv.style.display = 'block'; // Show the file name div

    // Preview the image
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function (e) {
        const imagePreview = document.getElementById('image-preview');
        imagePreview.src = e.target.result; // Set image source
        imagePreview.style.display = 'block'; // Show image preview
    }
    reader.readAsDataURL(file); // Read file as data URL
};

// Function to update the history dynamically
function updateHistory(resultData) {
    const entry = `
        <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
            <p><strong>Image:</strong> ${resultData.filename}</p>
            <p><strong>Disease Name:</strong> ${resultData.disease_name}</p>
            <p><strong>Fertilizer Recommendation:</strong> ${resultData.fertilizer}</p>
            <p><strong>Treatment Recommendation:</strong> ${resultData.treatment}</p>
        </div>
    `;
    historyItems.insertAdjacentHTML('afterbegin', entry); // Insert at the beginning to show the latest first
}
// Form submission to predict the disease
// Form submission to predict the disease
form.onsubmit = async (e) => {
    e.preventDefault(); // Prevent the default form submission
    diseaseNameSpan.innerText = ''; // Clear previous result
    fertilizerSpan.innerText = ''; // Clear previous result
    treatmentSpan.innerText = ''; // Clear previous result
    resultDiv.style.display = 'none'; // Hide result initially

    // Show the loading spinner
    spinner.style.display = 'block';
    document.getElementById('spinner-text').style.display = 'block';

    const formData = new FormData(form);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        // Hide the spinner once the response is received
        spinner.style.display = 'none';
        document.getElementById('spinner-text').style.display = 'none';

        // Check the response status and log
        console.log('Response status:', response.status);
        if (response.ok) {
            const resultData = await response.json(); // Expect JSON response
            console.log('Result data:', resultData); // Log the result to check

            diseaseNameSpan.innerText = resultData.disease_name; // Display the disease name
            fertilizerSpan.innerText = resultData.fertilizer; // Display the fertilizer
            treatmentSpan.innerText = resultData.treatment; // Display the treatment

            resultDiv.classList.add('show'); // Add class to fade in results
            resultDiv.style.display = 'block'; // Show result div

            // filenameLabel.innerText = 'Choose an image to upload'; // Reset label after prediction
            imagePreview.style.display = 'none'; // Hide image preview after prediction
            filename.style.display = 'none'; // Hide image preview after prediction

            // Dynamically update the history without refreshing
            updateHistory(resultData);

        } else {
            const errorMessage = await response.text(); // Capture response text for error
            console.error('Prediction error:', errorMessage);
            document.getElementById('error-message').innerText = 'Error in prediction. Please try again.';
            document.getElementById('error-message').style.display = 'block';
        }
    } catch (error) {
        // Catch any network errors or fetch exceptions
        console.error('Error during prediction:', error);
        document.getElementById('error-message').innerText = 'An error occurred during prediction. Please try again.';
        document.getElementById('error-message').style.display = 'block';

        // Hide spinner if error
        spinner.style.display = 'none';
        document.getElementById('spinner-text').style.display = 'none';
    }
};
// Clear button logic
const clearBtn = document.getElementById('clear-btn');
clearBtn.onclick = () => {
    form.reset(); // Reset the form
    //filenameLabel.innerText = 'Choose an image to upload'; // Reset label
    imagePreview.style.display = 'none'; // Hide image preview
    resultDiv.style.display = 'none'; // Hide results
};

// Get the history section and buttons
const historyBtn = document.getElementById('history-btn');
const historyItems = document.getElementById('history-items')
const historyList = document.getElementById('history-list');

// Function to reattach the event listeners
function reattachHistoryControls() {
    // Back button click logic
    const backBtn = document.getElementById('back-btn');
    backBtn.onclick = () => {
        historyList.style.display = 'none';
    };

    // Clear History Button click logic
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    clearHistoryBtn.onclick = async () => {
        try {
            const response = await fetch('/clear_history', {
                method: 'POST'
            });

            if (response.ok) {
                // After clearing, reset the history items inside the #history-items container
                historyItems.innerHTML = '<div class="centre"><p>History cleared successfully.</p></div>';
                // Re-attach the back button after clearing the history
                reattachHistoryControls();
            } else {
                console.error('Error clearing history:', response.status);
            }
        } catch (error) {
            console.error('Error clearing history:', error);
        }
    };
}

// Toggle visibility of the history section
historyBtn.onclick = async () => {
    const isHistoryVisible = historyList.style.display === 'block';

    // Toggle visibility of the history section
    if (isHistoryVisible) {
        historyList.style.display = 'none';
    } else {
        historyList.style.display = 'block';

        // Clear the history items container before fetching data
        historyItems.innerHTML = ''; // Clear history before fetching

        // Fetch and display the history if not already visible
        try {
            const response = await fetch('/history');
            const historyData = await response.json();

            

            if (historyData.length > 0) {
                historyData.forEach(entry => {
                    const item = document.createElement('div');
                    item.innerHTML = `
                        <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
                            <p><strong>Image:</strong> ${entry.filename}</p>
                            <p><strong>Disease Name:</strong> ${entry.disease_name}</p>
                            <p><strong>Fertilizer Recommendation:</strong> ${entry.fertilizer}</p>
                            <p><strong>Treatment Recommendation:</strong> ${entry.treatment}</p>
                        </div>
                    `;
                    historyItems.appendChild(item);
                });
            } else {
                historyItems.innerHTML = '<div class="centre"><p>No history available.</p></div>';
            }
            // Re-attach the event listeners for the back and clear buttons
            reattachHistoryControls();
            
            

        } catch (error) {
            console.error('Error fetching history:', error);
        }

        
    }
};

// Back button click logic
const backBtn = document.getElementById('back-btn');
backBtn.onclick = () => {
    historyList.style.display = 'none';
};

const clearHistoryBtn = document.getElementById('clear-history-btn');
// Clear History Button click logic 
clearHistoryBtn.onclick = async () => {
    try {
        const response = await fetch('/clear_history', {
            method: 'POST'
        });

        if (response.ok) {
            // After clearing, reset the history items inside the #history-items container
            historyItems.innerHTML = '<p>History cleared successfully.</p>';
        } else {
            console.error('Error clearing history:', response.status);
        }
    } catch (error) {
        console.error('Error clearing history:', error);
    }
};
