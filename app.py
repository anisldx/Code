import streamlit as st
import pandas as pd
import joblib
import requests
import json
from datetime import datetime
from abc import ABC, abstractmethod
from flight_delay_predictor import FlightDelayPredictor

class WeatherService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline'

    def fetch_weather_data(self, location: str, date: str) -> dict:
        url = f'{self.base_url}/{location}/{date}?key={self.api_key}&unitGroup=metric'
        response = requests.get(url)

        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                st.error(f'Error: Failed to parse JSON for {location}')
                return None
        else:
            st.error(f'Error: {response.status_code} for {location}')
            return None

    def extract_hourly_weather(self, weather_data: dict, hour: int) -> dict:
        if not weather_data or 'days' not in weather_data:
            return None

        for day in weather_data['days']:
            if 'hours' in day:
                for hour_data in day['hours']:
                    if hour_data['datetime'].startswith(f'{hour:02d}:'):
                        return {
                            'Temperature': hour_data.get('temp'),
                            'Humidity': hour_data.get('humidity'),
                            'Dew': hour_data.get('dew'),
                            'Precipitation': hour_data.get('precip'),
                            'Snow': hour_data.get('snow'),
                            'Snow_Depth': hour_data.get('snowdepth'),
                            'Wind_Speed': hour_data.get('windspeed'),
                            'Wind_Gust': hour_data.get('windgust'),
                            'Wind_Direction': hour_data.get('winddir'),
                            'Air_Pressure': hour_data.get('pressure'),
                            'Cloud_Cover': hour_data.get('cloudcover'),
                            'Visibility': hour_data.get('visibility'),
                        }
        return None

class AirportDataService:
    def __init__(self, data_file: str = "airport_movement_data.csv"):
        self.data_file = data_file
        self.city_map = {'GVA': 'Geneva', 'ZRH': 'Zurich'}

    def get_daily_departure_movement(self, departure_airport: str, departure_month: int, departure_day: int) -> int:
        city_name = self.city_map.get(departure_airport)
        if not city_name:
            st.error("Invalid departure airport code.")
            return None

        try:
            airport_data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            st.error(f"The file '{self.data_file}' was not found.")
            return None

        airport_data['Month'] = pd.to_datetime(airport_data['Date']).dt.month
        airport_data['Day'] = pd.to_datetime(airport_data['Date']).dt.day

        filtered_data = airport_data[
            (airport_data['City'] == city_name) &
            (airport_data['Month'] == departure_month) &
            (airport_data['Day'] == departure_day)
        ]

        if not filtered_data.empty:
            return filtered_data.iloc[0]['Departures_Count']
        else:
            st.warning(f"No data found for {city_name} on {departure_month}/{departure_day}.")
            return None

class FlightInputs:
    def __init__(self):
        self.route_distances = {
            ('GVA', 'LHR'): 754.67,
            ('GVA', 'FRA'): 459.76,
            ('GVA', 'CDG'): 407.6,
            ('ZRH', 'LHR'): 788.43,
            ('ZRH', 'FRA'): 285.43,
            ('ZRH', 'CDG'): 476.35
        }

    def is_valid_departure_hour(self, hour: int) -> bool:
        """Check if the departure hour is valid (between 6 and 0)."""
        return hour == 0 or hour >= 6

    def get_inputs(self, key_prefix: str = "") -> dict:
        inputs = {
            'aircraft_type': st.sidebar.selectbox(
                "Aircraft Type (IATA Code)", 
                ['223', '32Q', '32A', '320', '221', '32N', '295', '319', '321', 'CR9', 'E90', '290', '318', '32B', 'E95', 'CRK'],
                key=f'{key_prefix}aircraft_type'
            ),
            'departure_airport': st.sidebar.selectbox(
                "Departure Airport (IATA Code)", 
                ['GVA', 'ZRH'],
                key=f'{key_prefix}departure_airport'
            ),
            'arrival_airport': st.sidebar.selectbox(
                "Arrival Airport (IATA Code)", 
                ['CDG', 'FRA', 'LHR'],
                key=f'{key_prefix}arrival_airport'
            ),
            'flight_duration': st.sidebar.number_input(
                "Flight Duration (minutes)", 
                30, 500, 120,
                key=f'{key_prefix}flight_duration'
            ),
            'departure_date': st.sidebar.date_input(
                "Departure Date",
                value=pd.Timestamp("2025-01-01"),
                key=f'{key_prefix}departure_date'
            ),
            'departure_hour': st.sidebar.slider(
                "Departure Hour", 
                0, 23, 12,
                key=f'{key_prefix}departure_hour'
            ),
            'departure_day_of_week': st.sidebar.selectbox(
                "Day of Week", 
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                key=f'{key_prefix}day_of_week'
            ),
            'season': st.sidebar.selectbox(
                "Season", 
                ["Winter", "Spring", "Summer", "Autumn"],
                key=f'{key_prefix}season'
            ),
            'public_holiday': st.sidebar.selectbox(
                "Public Holiday", 
                ["Yes", "No"],
                key=f'{key_prefix}holiday'
            )
        }
        
        if not self.is_valid_departure_hour(inputs['departure_hour']):
            st.sidebar.error("⚠️ There are no flights scheduled between 01:00 and 05:59.")
        
        return inputs

    def get_distance(self, departure_airport: str, arrival_airport: str, key_prefix: str = "") -> float:
        distance = self.route_distances.get((departure_airport, arrival_airport))
        if distance is None:
            return st.sidebar.number_input(
                "Distance Great Circle (km) [Manual Input]", 
                50.0, 10000.0, 800.0,
                key=f'{key_prefix}distance_manual'
            )
        return distance

class UIRenderer:
    @staticmethod
    def render_header():
        st.title("Flight Delay Prediction System")
        st.markdown(""" 
        This model is trained on real historical flight data from a specific European airline. It predicts the departure status of flights from two Swiss airports to three major European hubs. The goal is to assist business stakeholders in making better decisions when introducing new flights due to high demand on these routes, adjusting schedules effectively, saving costs, and improving customer satisfaction.  
        
        Enter the flight information and click 'Predict' to see whether the flight is likely to be delayed or on time.
        """)
        st.markdown("<br>", unsafe_allow_html=True)

    @staticmethod
    def render_weather_data(weather_data: dict):
        rounded_data = {k: round(v, 2) if isinstance(v, float) else v for k, v in weather_data.items()}
        df = pd.DataFrame([rounded_data])
        st.subheader("Weather Data Collected")
        st.dataframe(df, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

    @staticmethod
    def render_prediction(prediction: int, probability: float):
        st.subheader("Flight Departure Status Prediction")
        status_color = "red" if prediction == 1 else "#afff33"
        status_text = "Delayed" if prediction == 1 else "On-Time"
        
        st.markdown(
            f"""
            <h2 style="color: {status_color}; font-size: 30px; text-align: left;"> {status_text}</h2>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Delay Probability: {probability[0]:.2%}")
        st.markdown("<br>", unsafe_allow_html=True)

    @staticmethod
    def render_flight_summary(flight_details: dict):
        st.subheader("Flight Details Summary")
        st.write(f"Route: {flight_details['departure_airport']} → {flight_details['arrival_airport']}")
        st.write(f"Departure Date: {flight_details['formatted_date']}")
        st.write(f"Departure Time: {flight_details['departure_hour']:02d}:00")
        st.write(f"Aircraft: {flight_details['aircraft_type']}")
        st.write(f"Flight Duration: {flight_details['flight_duration']} minutes")
        st.write(f"Great Circle Distance: {flight_details['distance_km']} km")
        st.write(f"Airport Daily Departure Movement: {flight_details['airport_movements']}")

    @staticmethod
    def render_map(departure_airport: str):
        map_file = f"{departure_airport.lower()}_map.html"
        try:
            with open(map_file, "r") as file:
                map_html = file.read()
            st.components.v1.html(map_html, height=600, scrolling=False)
        except FileNotFoundError:
            st.error(f"Map file {map_file} not found.")

class FlightDelayApp:
    def __init__(self):
        st.set_page_config(layout="wide")
        self.model = joblib.load("flight_delay_model.joblib")
        self.weather_service = WeatherService(api_key='PQUSUW8RN3LRLG8CKA2QRRDBW')
        self.airport_service = AirportDataService()
        self.flight_inputs = FlightInputs()
        self.ui = UIRenderer()

    def run(self):
        self.ui.render_header()
        st.sidebar.image("logo.png", use_container_width=True)
        st.sidebar.header("Flight Details")

        inputs = self.flight_inputs.get_inputs()
        distance_km = self.flight_inputs.get_distance(
            inputs['departure_airport'],
            inputs['arrival_airport']
        )

        airport_movements = self.airport_service.get_daily_departure_movement(
            inputs['departure_airport'],
            inputs['departure_date'].month,
            inputs['departure_date'].day
        )

        if airport_movements is None:
            st.error("Unable to fetch 'Airport_Daily_Departure_Movement'. Please check your input or the data.")
            return

        predict_button = st.sidebar.button("Predict", key='predict_button')
        
        if predict_button:
            if not self.flight_inputs.is_valid_departure_hour(inputs['departure_hour']):
                st.error("There are no flights scheduled for the chosen route between 01:00 and 05:59. Please select a departure time between 06:00 and 00:00.")
            else:
                self._handle_prediction(inputs, distance_km, airport_movements)

    def _handle_prediction(self, inputs: dict, distance_km: float, airport_movements: int):
        location = 'Geneva' if inputs['departure_airport'] == 'GVA' else 'Zurich'
        date = inputs['departure_date'].strftime('%Y-%m-%d')

        with st.spinner('Fetching weather data...'):
            weather_data = self._get_weather_data(location, date, inputs['departure_hour'])
            if weather_data:
                self.ui.render_weather_data(weather_data)

                prediction_input = self._prepare_prediction_input(
                    inputs, distance_km, airport_movements, weather_data
                )
                
                prediction = self.model.predict(prediction_input)
                probability = self.model.predict_proba(prediction_input)
                
                self.ui.render_prediction(prediction[0], probability)
                
                flight_summary = {
                    **inputs,
                    'distance_km': distance_km,
                    'airport_movements': airport_movements,
                    'formatted_date': inputs['departure_date'].strftime('%d/%m/%Y')
                }
                self.ui.render_flight_summary(flight_summary)
                
                st.markdown("<br>", unsafe_allow_html=True)
                self.ui.render_map(inputs['departure_airport'])

    def _get_weather_data(self, location: str, date: str, hour: int) -> dict:
        weather_response = self.weather_service.fetch_weather_data(location, date)
        if not weather_response:
            st.error("Failed to fetch weather data. Please try again.")
            return None
        
        weather_data = self.weather_service.extract_hourly_weather(weather_response, hour)
        if not weather_data:
            st.error("No weather data available for the specified time.")
            return None
            
        return weather_data

    def _prepare_prediction_input(self, inputs: dict, distance_km: float, airport_movements: int, weather_data: dict) -> pd.DataFrame:
        input_data = {
            'Aircraft_Type_IATA': inputs['aircraft_type'],
            'Flight_Duration': inputs['flight_duration'],
            'Departure_Airport_IATA': inputs['departure_airport'],
            'Arrival_Airport_IATA': inputs['arrival_airport'],
            'Distance_Great_Circle_Kilometers': distance_km,
            'Departure_Year': inputs['departure_date'].year,
            'Departure_Month': inputs['departure_date'].month,
            'Departure_Day_of_Month': inputs['departure_date'].day,
            'Departure_Day_of_Week': inputs['departure_day_of_week'],
            'Departure_Hour': inputs['departure_hour'],
            'Season': inputs['season'],
            'Public_Holiday': inputs['public_holiday'],
            'Airport_Daily_Departure_Movement': airport_movements,
            **weather_data
        }
        return pd.DataFrame([input_data])

if __name__ == "__main__":
    app = FlightDelayApp()
    app.run()
