// script.js for any dynamic functionality

document.addEventListener("DOMContentLoaded", function () {
    // Check for form submission
    const form = document.querySelector("form");
    
    if (form) {
        form.addEventListener("submit", function (e) {
            const fileInput = form.querySelector('input[type="file"]');
            
            if (!fileInput.files.length) {
                e.preventDefault();
                alert("Please upload a CSV file before submitting.");
            }
        });
    }
});
