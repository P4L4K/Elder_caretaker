document.getElementById('registerForm').addEventListener('submit', async (e) => {
    alert('Registering... Please wait.');
    e.preventDefault();
    
    try {
        // Get main caretaker details
        const formData = {
            full_name: document.getElementById('full_name').value,
            email: document.getElementById('email').value,
            username: document.getElementById('username').value,
            phone_number: document.getElementById('phone_number').value,
            password: document.getElementById('password').value,
            care_recipients: []
        };

        // Validate main caretaker details
        if (!formData.full_name || !formData.email || !formData.username || !formData.phone_number || !formData.password) {
            throw new Error('Please fill out all the main caretaker fields.');
        }

        const recipientDivs = document.querySelectorAll('.care-recipient');
        let validationError = null;
        recipientDivs.forEach(div => {
            if (validationError) return; // Stop processing if an error has been found

            const genderSelect = div.querySelector('[name="recipient_gender"]');
            const gender = genderSelect.value;
            
            if (!gender) {
                validationError = 'Please select a gender for all care recipients';
                return;
            }

            const recipient = {
                full_name: div.querySelector('[name="recipient_name"]').value,
                email: div.querySelector('[name="recipient_email"]').value,
                phone_number: div.querySelector('[name="recipient_phone"]').value,
                age: parseInt(div.querySelector('[name="recipient_age"]').value),
                gender: gender,
                respiratory_condition_status: div.querySelector('[name="recipient_condition"]').value === 'true'
            };

            // Validate the data
            if (!recipient.full_name) validationError = 'Full name is required for all care recipients';
            else if (!recipient.email) validationError = 'Email is required for all care recipients';
            else if (!recipient.phone_number || recipient.phone_number.length !== 10) validationError = 'Valid 10-digit phone number is required for all care recipients';
            else if (!recipient.age || isNaN(recipient.age)) validationError = 'Valid age is required for all care recipients';
            
            if (validationError) return;

            formData.care_recipients.push(recipient);
        });

        if (validationError) {
            throw new Error(validationError);
        }

        console.log('Sending registration data:', formData);
        const response = await fetch('http://localhost:8000/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        console.log('Registration response status:', response.status);
        const responseData = await response.json();
        console.log('Registration response data:', responseData);

        if (!response.ok) {
            alert(responseData.detail || 'Registration failed. Please try again.');
            return;
        }

        // Successful signup: receive token and created recipients; upload files if present
        const token = responseData.result && responseData.result.access_token;
        const createdRecipients = responseData.result && responseData.result.care_recipients ? responseData.result.care_recipients : [];

        // For each recipient form on the page, if a file was selected, upload it to the server
        const recipientDivsAfter = document.querySelectorAll('.care-recipient');
        for (let i = 0; i < recipientDivsAfter.length && i < createdRecipients.length; i++) {
            const div = recipientDivsAfter[i];
            const fileInput = div.querySelector('[name="recipient_report"]');
            if (fileInput && fileInput.files && fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const uploadForm = new FormData();
                uploadForm.append('file', file, file.name);
                try {
                    const uploadResp = await fetch(`http://localhost:8000/recipients/${createdRecipients[i].id}/reports`, {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + token
                        },
                        body: uploadForm
                    });
                    if (!uploadResp.ok) {
                        const err = await uploadResp.json();
                        console.warn('Failed to upload report for recipient', createdRecipients[i].id, err);
                    } else {
                        console.log('Uploaded report for recipient', createdRecipients[i].id);
                    }
                } catch (ue) {
                    console.error('Upload error:', ue);
                }
            }
        }

        alert('Registration successful! Please check your email for confirmation.');
        window.location.href = 'index.html';
    } catch (error) {
        console.error('Error details:', error);
        
        // Handle validation errors from the backend
        if (error.response && error.response.status === 422) {
            const errorData = await error.response.json();
            const errorMessage = errorData.detail[0].msg;
            alert('Validation error: ' + errorMessage);
        } else if (error.message) {
            // Handle frontend validation errors
            alert(error.message);
        } else {
            alert('An error occurred. Please check the browser console for details.');
        }
    }
});

// Add new care recipient form
document.getElementById('addRecipient').addEventListener('click', () => {
    const template = document.querySelector('.care-recipient').cloneNode(true);
    // Clear the values
    template.querySelectorAll('input, select').forEach(input => input.value = '');
    // Show remove button for additional recipients
    template.querySelector('.remove-recipient').style.display = 'flex';
    document.getElementById('careRecipients').appendChild(template);
});

// Handle remove recipient
document.getElementById('careRecipients').addEventListener('click', (e) => {
    if (e.target.classList.contains('remove-recipient') || 
        e.target.closest('.remove-recipient')) {
        const recipientDiv = e.target.closest('.care-recipient');
        recipientDiv.remove();
    }
});