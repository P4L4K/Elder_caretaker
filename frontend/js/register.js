document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get main caretaker details
    const formData = {
        full_name: document.getElementById('full_name').value,
        email: document.getElementById('email').value,
        username: document.getElementById('username').value,
        phone_number: document.getElementById('phone_number').value,
        password: document.getElementById('password').value,
        care_recipients: []
    };

    // Get care recipients details
    const recipientDivs = document.querySelectorAll('.care-recipient');
    recipientDivs.forEach(div => {
        const genderSelect = div.querySelector('[name="recipient_gender"]');
        const gender = genderSelect.value;
        
        if (!gender) {
            throw new Error('Please select a gender for all care recipients');
        }

        const recipient = {
            full_name: div.querySelector('[name="recipient_name"]').value,
            email: div.querySelector('[name="recipient_email"]').value,
            phone_number: div.querySelector('[name="recipient_phone"]').value,
            age: parseInt(div.querySelector('[name="recipient_age"]').value),
            gender: gender, // This will now be "Male", "Female", or "Other"
            respiratory_condition_status: div.querySelector('[name="recipient_condition"]').value === 'true'
        };

        // Validate the data
        if (!recipient.full_name) throw new Error('Full name is required for all care recipients');
        if (!recipient.email) throw new Error('Email is required for all care recipients');
        if (!recipient.phone_number || recipient.phone_number.length !== 10) throw new Error('Valid 10-digit phone number is required for all care recipients');
        if (!recipient.age || isNaN(recipient.age)) throw new Error('Valid age is required for all care recipients');
        
        formData.care_recipients.push(recipient);
    });

    try {
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

        const data = await response.json();

        if (response.ok) {
            alert('Registration successful! Please check your email for confirmation.');
            window.location.href = 'index.html';
        } else {
            alert(data.detail || 'Registration failed. Please try again.');
        }
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