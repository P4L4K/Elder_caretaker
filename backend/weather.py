import os
import requests
import datetime
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the backend directory
env_path = Path(__file__).parent / '../.env'
load_dotenv(env_path)

class WeatherPredictionModel:
    def __init__(self, api_key, city):
        self.api_key = api_key
        self.city = city
        # UPDATED: WeatherAPI.com Base URL
        self.base_url = "http://api.weatherapi.com/v1/forecast.json"
        self.headers = {'User-Agent': 'WeatherPredictor/2.0'}

        # UPDATED: Thresholds for Alerts (US EPA Index 1-6 used by WeatherAPI)
        self.AQI_THRESHOLDS = {
            1: ("Good", "No risk. Air quality is satisfactory."),
            2: ("Moderate", "Acceptable. Sensitive groups may feel effects."),
            3: ("Unhealthy for Sensitive Groups", "General public OK, sensitive groups should reduce exertion."),
            4: ("Unhealthy", "Some public health effects. Sensitive groups avoid outdoor activities."),
            5: ("Very Unhealthy", "Health Alert. Risk for everyone."),
            6: ("Hazardous", "Health Warning. Emergency conditions.")
        }
        self.TEMP_HIGH_ALERT = 35.0  # Celsius
        self.TEMP_LOW_ALERT = 0.0    # Celsius
        self.HUMIDITY_HIGH_ALERT = 85 # Percent
        self.HUMIDITY_LOW_ALERT = 20  # Percent

    def fetch_data(self):
        """
        Fetches Current, Forecast, AQI, and Alerts in a SINGLE call.
        WeatherAPI.com structure allows this via the 'days' and 'aqi' params.
        """
        params = {
            'key': self.api_key,
            'q': self.city,
            'days': 3,       # Get today + next 2 days to ensure we have full 24h data
            'aqi': 'yes',    # Request Air Quality Data
            'alerts': 'yes'  # Request Official Government Alerts
        }

        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                print("Error: Location not found or invalid API Key.")
            elif response.status_code == 401:
                print("Error: Unauthorized. Check your WeatherAPI.com API Key.")
            else:
                print(f"HTTP Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Connection Error: {e}")
            sys.exit(1)

    def analyze_conditions(self, data):
        """Parses WeatherAPI.com JSON structure for current conditions."""

        current = data['current']

        # Extract Parameters (WeatherAPI uses 'temp_c', 'humidity', 'air_quality')
        temp = current['temp_c']
        humidity = current['humidity']

        # AQI is nested in 'air_quality' -> 'us-epa-index' (1-6 scale)
        # Default to 0 if missing
        aqi_index = current.get('air_quality', {}).get('us-epa-index', 0)

        aqi_status, aqi_advice = self.AQI_THRESHOLDS.get(aqi_index, ("Unknown", "No Data Available"))

        analysis = {
            "temperature": temp,
            "humidity": humidity,
            "aqi_index": aqi_index,
            "aqi_status": aqi_status,
            "condition_text": current['condition']['text'],
            "comfort_status": "Optimal"
        }

        # Calculate Comfort Status (Simple Heat Index Proxy)
        if temp > 30 and humidity > 70:
            analysis['comfort_status'] = "Oppressive Heat (Sticky)"
        elif temp < 10 and humidity > 80:
            analysis['comfort_status'] = "Damp Cold"
        elif aqi_index >= 4:
            analysis['comfort_status'] = "Toxic Air Quality"
        elif 20 <= temp <= 25 and 30 <= humidity <= 60:
            analysis['comfort_status'] = "Perfect Comfort"

        return analysis

    def generate_alerts(self, analysis, api_alerts):
        """Generates strict alerts based on parameters + Official API Alerts."""
        alerts = []

        # 1. Official Government Alerts (from API)
        if api_alerts:
            for alert in api_alerts.get('alert', []):
                alerts.append(f"[OFFICIAL] {alert['event']}: {alert['headline']}")

        # 2. Custom AQI Alerts
        if analysis['aqi_index'] >= 3:
            severity = "CRITICAL" if analysis['aqi_index'] >= 4 else "WARNING"
            alerts.append(f"[{severity}] AQI is {analysis['aqi_status']} ({analysis['aqi_index']}/6).")

        # 3. Temperature Alerts
        if analysis['temperature'] >= self.TEMP_HIGH_ALERT:
            alerts.append(f"[DANGER] Heatwave conditions ({analysis['temperature']}°C). Stay hydrated.")
        elif analysis['temperature'] <= self.TEMP_LOW_ALERT:
            alerts.append(f"[WARNING] Freezing conditions ({analysis['temperature']}°C). Risk of frost.")

        # 4. Humidity Alerts
        if analysis['humidity'] > self.HUMIDITY_HIGH_ALERT:
            alerts.append("[ADVISORY] High Humidity. Mold risk and reduced sweat evaporation.")
        elif analysis['humidity'] < self.HUMIDITY_LOW_ALERT:
            alerts.append("[ADVISORY] Very Dry Air. Fire risk increased; hydrate skin.")

        if not alerts:
            alerts.append("[OK] No severe weather alerts at this time.")

        return alerts

    def predict_next_24h(self, forecast_data):
        """
        Predicts trend by splicing hourly data from 'Today' and 'Tomorrow'
        to get the immediate next 24 hours.
        """
        # WeatherAPI returns 'forecastday' list. Each has an 'hour' list.
        # We combine hours from today and tomorrow to find the next 24h block.

        hourly_data = []
        # Add today's remaining hours and tomorrow's hours
        for day in forecast_data['forecastday']:
            hourly_data.extend(day['hour'])

        # Find current hour index (approximate based on system time)
        # Note: In a production app, we would parse 'time_epoch' vs current time.
        # Here we just take the next 24 entries assuming the API returns past hours for today.

        current_epoch = datetime.datetime.now().timestamp()

        # Filter: Only keep hours in the future
        future_hours = [h for h in hourly_data if h['time_epoch'] > current_epoch]

        # Take the next 24 hours (or whatever is left)
        next_24h = future_hours[:24]

        if not next_24h:
            return {"error": "Insufficient forecast data"}

        temps = [item['temp_c'] for item in next_24h]
        # WeatherAPI uses 'chance_of_rain' (0-100) directly in the hour object
        rain_chances = [item['chance_of_rain'] for item in next_24h]
        max_rain_prob = max(rain_chances) if rain_chances else 0

        trend = "Stable"
        if temps[-1] > temps[0] + 3:
            trend = "Warming Up"
        elif temps[-1] < temps[0] - 3:
            trend = "Cooling Down"

        return {
            "max_temp": max(temps),
            "min_temp": min(temps),
            "precip_chance": f"{max_rain_prob}%",
            "trend": trend
        }

    def run(self):
        print(f"--- Weather Prediction Model Initialized for {self.city} ---")
        print("Fetching reliable data from WeatherAPI.com (Accuracy Target: >95%)...")

        full_data = self.fetch_data()

        # Process Data
        analysis = self.analyze_conditions(full_data)
        alerts = self.generate_alerts(analysis, full_data.get('alerts', {}))
        prediction = self.predict_next_24h(full_data['forecast'])

        # --- OUTPUT DASHBOARD ---
        print("\n" + "="*40)
        print(f"CURRENT REPORT: {self.city.upper()}")
        print("="*40)
        print(f"🌡️  Temperature : {analysis['temperature']}°C")
        print(f"☁️  Condition   : {analysis['condition_text']}")
        print(f"💧 Humidity    : {analysis['humidity']}%")
        print(f"🌫️  AQI Index   : {analysis['aqi_index']} - {analysis['aqi_status']}")
        print(f"🧠 Analysis    : {analysis['comfort_status']}")
        print("-" * 40)

        print("\n📢 ACTIVE ALERTS:")
        for alert in alerts:
            print(f"  > {alert}")

        print("\n🔮 24-HOUR PREDICTION MODEL:")
        print(f"  • Trend           : {prediction['trend']}")
        print(f"  • Max Temp        : {prediction['max_temp']}°C")
        print(f"  • Min Temp        : {prediction['min_temp']}°C")
        print(f"  • Rain Probability: {prediction['precip_chance']}")
        print("="*40)

# --- EXECUTION ---
if __name__ == "__main__":
    API_KEY = os.getenv('WEATHER_API_KEY')
    if not API_KEY:
        print("Error: WEATHER_API_KEY not found in .env file")
        sys.exit(1)
    city_input = "Jammu"
    model = WeatherPredictionModel(API_KEY, city_input)
    model.run()