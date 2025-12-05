// e:\model_test\caretaker\frontend\js\weatherWidget.js
class WeatherWidget {
    constructor(containerId, apiBaseUrl) {
        this.containerId = containerId;
        this.API_BASE_URL = apiBaseUrl || 'http://127.0.0.1:8000/api';
        this.TOKEN = localStorage.getItem('token');
        this.init();
    }

    // Initialize the widget
    init() {
        this.createWidget();
        this.updateTime();
        this.updateWeatherDisplay();
        
        // Update time every minute
        setInterval(() => this.updateTime(), 60000);
        
        // Update weather every 5 minutes
        setInterval(() => this.updateWeatherDisplay(), 300000);
    }

    // Create the widget HTML
    createWidget() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="weather-widget" style="
                background: linear-gradient(135deg, rgba(17, 24, 39, 0.8), rgba(17, 24, 39, 0.9));
                border-radius: 16px;
                padding: 20px;
                border: 1px solid rgba(56, 189, 248, 0.2);
                margin-bottom: 20px;
            ">
                <div style="text-align: center; margin-bottom: 15px;">
                    <h3 style="margin: 0; font-size: 1.2rem; font-weight: 600;" id="location">Loading...</h3>
                    <p id="current-time" style="margin: 4px 0 0; color: #94a3b8; font-size: 0.9rem;">--:-- --</p>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8;" id="tempValue">--°C</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Temperature</div>
                    </div>
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8;" id="humidityValue">--%</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Humidity</div>
                    </div>
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8;" id="aqiValue">--</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">AQI</div>
                    </div>
                </div>
            </div>
        `;
    }

    // Update time display
    updateTime() {
        const now = new Date();
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            timeElement.textContent = now.toLocaleString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }
    }

    // Fetch weather data
    async fetchWeatherData() {
        try {
            const response = await fetch(`${this.API_BASE_URL}/weather/current`, {
                headers: {
                    'Authorization': `Bearer ${this.TOKEN}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch weather data');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching weather data:', error);
            return null;
        }
    }

    // Update weather display
    async updateWeatherDisplay() {
        const data = await this.fetchWeatherData();
        if (!data) {
            console.log('No weather data available');
            return;
        }

        // Update location
        if (document.getElementById('location')) {
            document.getElementById('location').textContent = data.location || 'Unknown';
        }

        // Update stats
        const updateElement = (id, value, suffix = '') => {
            const element = document.getElementById(id);
            if (element) element.textContent = value + suffix;
        };

        updateElement('tempValue', data.temperature, '°C');
        updateElement('humidityValue', data.humidity, '%');
        updateElement('aqiValue', data.aqi);
    }
}

// Export the WeatherWidget class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WeatherWidget;
}