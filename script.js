document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const resultsDiv = document.getElementById('results');

    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission

        const fileInput = document.querySelector('input[type="file"]');
        const file = fileInput.files[0];

        if (!file) {
            alert('Please select a resume file to upload.');
            return;
        }

        // Display loading message
        resultsDiv.innerHTML = '<p>Analyzing your resume, please wait...</p>';

        // Create FormData object to send the file
        const formData = new FormData();
        formData.append('resume', file);

        // Send the file to the server using fetch
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display the results
            if (data.success) {
                resultsDiv.innerHTML = `<h2>Analysis Results:</h2><p>${data.message}</p>`;
                console.log(data.message);
            } else {
                resultsDiv.innerHTML = `<h2>Error: This is error</h2><p>${data.error}</p>`;
                console.log(data.error);
            }
        })
        .catch(error => {
            resultsDiv.innerHTML = `<h2>Error:</h2><p>There was a problem with the request.</p>`;
            console.error('Error:', error);
        });
    });
});