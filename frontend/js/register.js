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
        const API_BASE = 'http://127.0.0.1:8000';
        const response = await fetch(API_BASE + '/api/signup', {
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
        // Save token so the user is logged in after registration
        if (token) {
            localStorage.setItem('token', token);
        }
        const createdRecipients = responseData.result && responseData.result.care_recipients ? responseData.result.care_recipients : [];

        console.log('Signup createdRecipients:', createdRecipients);
        // For each recipient form on the page, if a file was selected, upload it to the server
        const recipientDivsAfter = document.querySelectorAll('.care-recipient');
        async function fileToBase64(file) {
            return await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const result = reader.result || '';
                    // result is like 'data:<mime>;base64,AAAA...'
                    const parts = result.split(',');
                    resolve(parts.length > 1 ? parts[1] : parts[0]);
                };
                reader.onerror = (e) => reject(e);
                reader.readAsDataURL(file);
            });
        }

       // Replace the entire file upload block (lines 102-131) with this:
for (let i = 0; i < recipientDivsAfter.length && i < createdRecipients.length; i++) {
    const div = recipientDivsAfter[i];
    const fileInput = div.querySelector('[name="recipient_report"]');
    
    if (fileInput && fileInput.files && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const uploadResp = await fetch(
                `${API_BASE}/api/recipients/${createdRecipients[i].id}/reports`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    },
                    body: formData
                }
            );

            if (!uploadResp.ok) {
                const errorData = await uploadResp.json().catch(() => ({}));
                console.error('Upload failed:', uploadResp.status, errorData);
                throw new Error(`Failed to upload report: ${errorData.detail || 'Unknown error'}`);
            }
            
            const result = await uploadResp.json();
            console.log('Upload successful:', result);
            
        } catch (error) {
            console.error('Upload error:', error);
            throw new Error(`Failed to upload medical report: ${error.message}`);
        }
    }
}

        alert('Registration successful! You are now logged in. Redirecting to dashboard.');
        window.location.href = 'profile.html';
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