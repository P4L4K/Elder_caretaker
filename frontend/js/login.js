document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        console.log('Sending login data:', { username });
        const response = await fetch('http://localhost:8000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username,
                password
            })
        });
        
        console.log('Login response status:', response.status);

        const data = await response.json();

        if (response.ok) {
            // Store the token
            localStorage.setItem('token', data.result.access_token);
            // Redirect to the redesigned profile landing page
            window.location.href = 'profile.html';
        } else {
            alert(data.detail || 'Login failed. Please try again.');
        }
    } catch (error) {
        console.error('Error details:', error);
        alert('An error occurred. Please check the browser console for details.');
    }
});