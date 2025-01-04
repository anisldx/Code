#############################################################################################
#
# This program processes flight, weather, and movement data for GVA and ZRH airports. 
# It merges flight data with weather data, movement data, and create new features to enrich insights.
# The processed data is stored in a CSV file, ready for further analysis (EDA).
# 
# Key Features:
#  - Cleans, enriches, and merges data from different sources.
#  - Estimates delay costs based on EUROCONTROL recommended values.
#  - Modular design with a class-based approach for scalability.
#
# Prepared by: Anis Larid
# Created on: November 26, 2024
# Version: 1.0
#
#############################################################################################

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

class FlightDataProcessor:
    def __init__(self, flight_filepath, weather_filepath, movement_filepath):
        self.flight_filepath = flight_filepath
        self.weather_filepath = weather_filepath
        self.movement_filepath = movement_filepath
        self.flight_df = pd.DataFrame()
        self.weather_df = pd.DataFrame()
        self.movement_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()

    def load_data(self):
        """Load flight data, weather data, and movement data from CSV files."""
        try:
            self.flight_df = pd.read_csv(self.flight_filepath)
            logging.info(f"Loaded {len(self.flight_df)} rows from the flight dataset.")
        except FileNotFoundError:
            logging.error(f"Flight data file not found at {self.flight_filepath}.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            logging.error("Flight data file is empty.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An error occurred while loading flight data: {e}")
            sys.exit(1)

        try:
            self.weather_df = pd.read_csv(self.weather_filepath)
            logging.info(f"Loaded {len(self.weather_df)} rows from the weather dataset.")
        except FileNotFoundError:
            logging.error(f"Weather data file not found at {self.weather_filepath}.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            logging.error("Weather data file is empty.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An error occurred while loading weather data: {e}")
            sys.exit(1)

        try:
            self.movement_df = pd.read_csv(self.movement_filepath)
            logging.info(f"Loaded {len(self.movement_df)} rows from the airport movement dataset.")
        except FileNotFoundError:
            logging.error(f"Movement data file not found at {self.movement_filepath}.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            logging.error("Movement data file is empty.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An error occurred while loading movement data: {e}")
            sys.exit(1)

    def clean_flight_data(self):
        """Clean and preprocess the flight data."""
        # Drop rows with critical missing values
        critical_columns = [
            'elapsedTime', 'departure.airport.iata', 'departure.country.code', 
            'arrival.airport.iata', 'arrival.country.code', 'serviceType.iata',
            'distance.accumulatedGreatCircleKilometers', 'distance.greatCircleKilometers',
            'departure.estimatedTime.outGate.local', 'departure.actualTime.outGateTimeliness',
            'departure.actualTime.outGate.local', 'departure.actualTime.offGround.local'
        ]
        self.flight_df.dropna(subset=critical_columns, inplace=True)
        logging.info(f"Flight data after dropping rows with missing critical values: {len(self.flight_df)} rows.")

        # Drop duplicates
        self.flight_df.drop_duplicates(inplace=True)
        logging.info(f"Flight data after dropping duplicates: {len(self.flight_df)} rows.")

        # Rename columns for clarity
        self.flight_df.rename(columns={
            'actualAircraftType.iata': 'Aircraft_Type_IATA',
            'elapsedTime': 'Flight_Duration',
            'departure.airport.iata': 'Departure_Airport_IATA',
            'departure.country.code': 'Departure_Country_Code',
            'arrival.airport.iata': 'Arrival_Airport_IATA',
            'arrival.country.code': 'Arrival_Country_Code',
            'serviceType.iata': 'Service_Type_IATA',
            'distance.accumulatedGreatCircleKilometers': 'Distance_Accumulated_Great_Circle_Kilometers',
            'distance.greatCircleKilometers': 'Distance_Great_Circle_Kilometers',
            'departure.estimatedTime.outGate.local': 'Departure_Estimated_Local_Time_Out_Gate',
            'departure.actualTime.outGateTimeliness': 'Departure_Out_Gate_Status',
            'departure.actualTime.outGate.local': 'Departure_Actual_Local_Time_Out_Gate',
            'departure.actualTime.offGround.local': 'Departure_Actual_Local_Time_Off_Ground',
        }, inplace=True)

        # Drop rows where 'Departure_Actual_Local_Time_Off_Ground' is earlier than 'Departure_Actual_Local_Time_Out_Gate'
        self.flight_df = self.flight_df[self.flight_df['Departure_Actual_Local_Time_Off_Ground'] >= self.flight_df['Departure_Actual_Local_Time_Out_Gate']]
        logging.info(f"Flight data after removing anomalies: {len(self.flight_df)} rows.")

        # Aircraft types with fewer than 10 rows will be dropped
        threshold = 10  # Set a threshold for minimum representation

        # Calculate value counts
        aircraft_counts = self.flight_df['Aircraft_Type_IATA'].value_counts()

        # Filter the dataset to drop rare categories
        self.flight_df = self.flight_df[self.flight_df['Aircraft_Type_IATA'].isin(aircraft_counts[aircraft_counts >= threshold].index)]
        logging.info(f"Flight data after removing rare aircrafts: {len(self.flight_df)} rows.")

        # Convert datetime columns to timezone-aware datetime objects
        datetime_columns = [
            'Departure_Estimated_Local_Time_Out_Gate',
            'Departure_Actual_Local_Time_Out_Gate',
            'Departure_Actual_Local_Time_Off_Ground'
        ]
        for col in datetime_columns:
            self.flight_df[col] = pd.to_datetime(
                self.flight_df[col], utc=True, errors='coerce'
            ).dt.tz_convert('Europe/Zurich')

        # Remove rows with invalid datetime conversions
        self.flight_df.dropna(subset=datetime_columns, inplace=True)

        # Extract month, day of month, day of week, and hour
        self.flight_df['Departure_Month'] = self.flight_df['Departure_Estimated_Local_Time_Out_Gate'].dt.month
        self.flight_df['Departure_Day_of_Month'] = self.flight_df['Departure_Estimated_Local_Time_Out_Gate'].dt.day
        self.flight_df['Departure_Day_of_Week'] = self.flight_df['Departure_Estimated_Local_Time_Out_Gate'].dt.day_name()
        self.flight_df['Departure_Hour'] = self.flight_df['Departure_Estimated_Local_Time_Out_Gate'].dt.hour

        # Add a feature related to meteorological seasons
        def get_season(timestamp):
            month = timestamp.month
            if month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            elif month in [9, 10, 11]:
                return 'Autumn'
            else:
                return 'Winter'

        self.flight_df['Season'] = self.flight_df['Departure_Estimated_Local_Time_Out_Gate'].apply(get_season)

        # Prepare public holiday lists for Geneva and Zurich
        geneva_holidays = [pd.to_datetime(date).date() for date in [
        '2024-01-01', # New Year's Day
        '2024-03-29', # Good Friday
        '2024-04-01', # Easter Monday
        '2024-05-09', # Ascension Day
        '2024-05-20', # Whit Monday (Pentecost Monday)
        '2024-08-01', # National Day
        '2024-09-05', # Geneva Fast
        '2023-12-25', # Christmas Day                               # Update as soon as December 2024 Data is Available
        '2023-12-31', # Restoration of the Republic                 # Update as soon as December 2024 Data is Available
        ]]
        zurich_holidays = [pd.to_datetime(date).date() for date in [
        '2024-01-01', # New Year's Day
        '2024-01-02', # Berchtold's Day
        '2024-03-29', # Good Friday
        '2024-04-01', # Easter Monday
        '2024-05-01', # Labor Day
        '2024-05-09', # Ascension Day
        '2024-05-20', # Whit Monday (Pentecost Monday)
        '2024-08-01', # National Day
        '2023-12-25', # Christmas Day                               # Update as soon as December 2024 Data is Available
        '2023-12-26', # St. Stephen's Day                           # Update as soon as December 2024 Data is Available
        ]]

        # Function to determine if a flight is on a public holiday
        def is_public_holiday(row):
            flight_date = row['Departure_Estimated_Local_Time_Out_Gate'].date()
            if row['Departure_Airport_IATA'] == 'GVA':
                return flight_date in geneva_holidays
            elif row['Departure_Airport_IATA'] == 'ZRH':
                return flight_date in zurich_holidays
            else:
                return False

        # Apply the function to create the 'Public_Holiday' column
        self.flight_df['Public_Holiday'] = self.flight_df.apply(is_public_holiday, axis=1)

        # Convert 'Public_Holiday' to a categorical type with labels
        self.flight_df['Public_Holiday'] = self.flight_df['Public_Holiday'].map({True: 'Yes', False: 'No'})

        # Calculate At Gate Delay
        self.flight_df['At_Gate_Delay'] = (
            (self.flight_df['Departure_Actual_Local_Time_Out_Gate'] -
             self.flight_df['Departure_Estimated_Local_Time_Out_Gate']).dt.total_seconds() / 60
        ).astype(float)

        # Calculate Taxiing Time
        self.flight_df['Taxiing_Out_Time'] = (
            (self.flight_df['Departure_Actual_Local_Time_Off_Ground'] -
             self.flight_df['Departure_Actual_Local_Time_Out_Gate']).dt.total_seconds() / 60
        ).astype(float)

        # Standard Taxiing Times by Airport according to EUROCONTROL
        standard_taxi_times = {
            'GVA': 10.19,
            'ZRH': 11.54
        }

        # Calculate Taxiing Out Delay
        self.flight_df['Taxiing_Out_Delay'] = self.flight_df.apply(
            lambda row: max(0, row['Taxiing_Out_Time'] - standard_taxi_times.get(row['Departure_Airport_IATA'], 0)),
            axis=1
        )

        # Calculate Total Delay Time
        self.flight_df['Total_Delay_Time'] = self.flight_df['At_Gate_Delay'] + self.flight_df['Taxiing_Out_Delay']

        # Round delay columns to 2 decimal points
        delay_columns = ['At_Gate_Delay', 'Taxiing_Out_Delay', 'Total_Delay_Time']
        self.flight_df[delay_columns] = self.flight_df[delay_columns].round(2)

        # Convert Flight Duration to float
        self.flight_df['Flight_Duration'] = pd.to_numeric(self.flight_df['Flight_Duration'], errors='coerce')

        # Remove rows with invalid flight duration
        self.flight_df.dropna(subset=['Flight_Duration'], inplace=True)

        # Classify flights into binary categories based on their Total_Delay_Time:
        # According to U.S. Department of Transportation (DOT) standards, a flight is considered “on-time”
        # if it departs (or arrives) within 15 minutes of the scheduled time. (Reference: Air Travel Consumer Report,
        # U.S. DOT: https://www.transportation.gov/airconsumer/air-travel-consumer-reports)
        #
        # Moreover, the EUROCONTROL recommended values for short delays (under 30 minutes) suggest that 
        # the cost impact is minimal. Therefore, having binary categories (e.g., < 15 minutes for 'On-Time' 
        # and > 15 minutes for 'Delayed') aligns with both industry-standard definitions and cost structures. 
        # This approach ensures that analysis and modeling efforts remain focused on operationally
        # delays while maintaining credibility and comparability with recognized aviation benchmarks.
        #
        # Categories:
        # - OnTime: < 15 minutes
        # - Delayed: > 15 minutes

        def classify_delay(x):
            if x < 15:
                return 'On-Time'
            else:
                return 'Delayed'
            
        # Apply the classification function to create a categorical Departure_Status column
        self.flight_df['Departure_Status'] = self.flight_df['Total_Delay_Time'].apply(classify_delay)

        # Round the Estimated Departure Time to the nearest hour
        self.flight_df['Rounded_Departure_Estimated_Local_Time_Out_Gate'] = self.flight_df[
            'Departure_Estimated_Local_Time_Out_Gate'].dt.round('h')

        # Create a Departure City feature
        airport_city_map = {'GVA': 'Geneva', 'ZRH': 'Zurich'}
        self.flight_df['Departure_City'] = self.flight_df['Departure_Airport_IATA'].map(airport_city_map)

        # Create 'At_Gate_Delay_Cost' and 'Taxiing_Out_Delay_Cost' features
        def calculate_delay_cost(total_delay_time, delay_time, short_cost, long_cost):
            if total_delay_time > 0 and total_delay_time < 30:
                return delay_time * short_cost
            elif total_delay_time >= 30:
                return delay_time * long_cost
            else:
                return 0

        self.flight_df['At_Gate_Delay_Cost'] = self.flight_df.apply(
            lambda row: max(0, calculate_delay_cost(row['Total_Delay_Time'], row['At_Gate_Delay'], 45, 166)), axis=1
        )

        self.flight_df['Taxiing_Out_Delay_Cost'] = self.flight_df.apply(
            lambda row: max(0, calculate_delay_cost(row['Total_Delay_Time'], row['Taxiing_Out_Delay'], 62, 182)), axis=1
        )

        # Calculate Total Delay Cost
        self.flight_df['Total_Delay_Cost'] = self.flight_df['At_Gate_Delay_Cost'] + self.flight_df['Taxiing_Out_Delay_Cost']

        # Round cost columns to 2 decimal places
        cost_columns = ['At_Gate_Delay_Cost', 'Taxiing_Out_Delay_Cost', 'Total_Delay_Cost']
        self.flight_df[cost_columns] = self.flight_df[cost_columns].round(2)

        # Define the coordinates for the departure airports
        departure_airport_coords = {
            'GVA': {'latitude': 46.238098, 'longitude': 6.10895},
            'ZRH': {'latitude': 47.458056, 'longitude': 8.548056}
        }

        # Map the latitude and longitude to the DataFrame for departure airports
        self.flight_df['Departure_Airport_Latitude'] = self.flight_df['Departure_Airport_IATA'].map(
            lambda x: departure_airport_coords.get(x, {}).get('latitude', np.nan)
        )
        self.flight_df['Departure_Airport_Longitude'] = self.flight_df['Departure_Airport_IATA'].map(
            lambda x: departure_airport_coords.get(x, {}).get('longitude', np.nan)
        )

        # Define the coordinates for the arrival airports
        arrival_airport_coords = {
        'LHR': {'latitude': 51.4706, 'longitude': -0.461941},
        'CDG': {'latitude': 49.012798, 'longitude': 2.55},
        'FRA': {'latitude': 50.030241, 'longitude': 8.561096},
        }

        # Map the latitude and longitude to the DataFrame for arrival airports
        self.flight_df['Arrival_Airport_Latitude'] = self.flight_df['Arrival_Airport_IATA'].map(
        lambda x: arrival_airport_coords.get(x, {}).get('latitude', np.nan)
        )
        self.flight_df['Arrival_Airport_Longitude'] = self.flight_df['Arrival_Airport_IATA'].map(
        lambda x: arrival_airport_coords.get(x, {}).get('longitude', np.nan)
        )

        # Drop unnecessary columns
        columns_to_drop = [
            'flightNumber', 'flightType', 'carrier.iata', 'Service_Type_IATA',
            'Distance_Accumulated_Great_Circle_Kilometers', 'Departure_Out_Gate_Status',
            'departure.actualTime.outGateVariation', 'arrival.actualTime.inGateTimeliness',
            'arrival.actualTime.inGateVariation', 'arrival.actualTime.onGround.local',
            'arrival.actualTime.inGate.local', 'departure.airport.iata.1',
            'departure.country.code.1', 'arrival.airport.iata.1', 'arrival.country.code.1',
            'Taxiing_Out_Time'
        ]
        existing_columns_to_drop = [col for col in columns_to_drop if col in self.flight_df.columns]
        self.flight_df.drop(columns=existing_columns_to_drop, inplace=True)

        # Reset index
        self.flight_df.reset_index(drop=True, inplace=True)
        logging.info("Flight data cleaning completed.")

    def clean_weather_data(self):
        """Clean and preprocess the weather data."""
        # Rename columns for clarity
        self.weather_df.rename(columns={
            'temp': 'Temperature',
            'humidity': 'Humidity',
            'precip': 'Precipitation',
            'dew': 'Dew',
            'preciptype': 'Precipitation_Type',
            'snow': 'Snow',
            'snowdepth': 'Snow_Depth',
            'windspeed': 'Wind_Speed',
            'windgust': 'Wind_Gust',
            'winddir': 'Wind_Direction',
            'pressure': 'Air_Pressure',
            'cloudcover': 'Cloud_Cover',
            'visibility': 'Visibility'
        }, inplace=True)

        # Fill missing preciptype values and clean the data
        self.weather_df['Precipitation_Type'] = self.weather_df['Precipitation_Type'].fillna("").astype(str)
        self.weather_df['Precipitation_Type'] = self.weather_df['Precipitation_Type'].str.replace(
            r"[\[\]']", "", regex=True).str.strip()

        # Update preciptype based on conditions if empty
        conditions = self.weather_df['conditions'].str.lower().fillna('')
        self.weather_df['Precipitation_Type'] = np.where(
            (conditions.str.contains('rain')) & (self.weather_df['Precipitation_Type'] == ""),
            'rain',
            self.weather_df['Precipitation_Type']
        )
        self.weather_df['Precipitation_Type'] = np.where(
            (conditions.str.contains('snow')) & (self.weather_df['Precipitation_Type'] == ""),
            'snow',
            self.weather_df['Precipitation_Type']
        )

        # Replace empty preciptype with 'none'
        self.weather_df['Precipitation_Type'] = self.weather_df['Precipitation_Type'].replace("", "none")

        # Forward fill missing values in the dataset
        self.weather_df.ffill(inplace=True)

        # Convert datetime column to timezone-aware datetime object
        self.weather_df['datetime_local'] = pd.to_datetime(
            self.weather_df['datetime_local'], utc=True, errors='coerce'
        ).dt.tz_convert('Europe/Zurich')

        # Remove rows with invalid datetime conversions
        self.weather_df.dropna(subset=['datetime_local'], inplace=True)

        # Drop unnecessary columns
        columns_to_drop = ['conditions', 'feelslike']
        existing_columns_to_drop = [col for col in columns_to_drop if col in self.weather_df.columns]
        self.weather_df.drop(columns=existing_columns_to_drop, inplace=True)
        logging.info("Weather data cleaning completed.")

    def clean_movement_data(self):
        """Clean and preprocess the airport movement data."""
        # Ensure date column is in datetime format
        self.movement_df['Date'] = pd.to_datetime(self.movement_df['Date'], errors='coerce')

        # Drop rows with invalid dates
        self.movement_df.dropna(subset=['Date'], inplace=True)

        # Drop duplicates
        self.movement_df.drop_duplicates(inplace=True)
        logging.info(f"Cleaned airport movement data contains {len(self.movement_df)} rows.")

    def merge_data(self):
        """Merge flight, weather, and movement data on keys."""
        # Merge flight and weather data
        self.merged_df = pd.merge(
            self.flight_df,
            self.weather_df,
            how='inner',
            left_on=['Rounded_Departure_Estimated_Local_Time_Out_Gate', 'Departure_City'],
            right_on=['datetime_local', 'city']
        )

        # Create a 'Date' column in the merged data for matching with 'airport_movement_data'
        self.merged_df['Date'] = self.merged_df['Rounded_Departure_Estimated_Local_Time_Out_Gate'].dt.date

         # Ensure both Date columns are datetime64[ns]
        self.merged_df['Date'] = pd.to_datetime(self.merged_df['Date'], errors='coerce')
        self.movement_df['Date'] = pd.to_datetime(self.movement_df['Date'], errors='coerce')

        # Merge with airport movement data using 'Date' and 'City'
        self.merged_df = pd.merge(
        self.merged_df,
        self.movement_df,
        how='left',
        left_on=['Date', 'Departure_City'],
        right_on=['Date', 'City']
    )
        
        # Rename 'Departures_Count' to 'Airport_Daily_Departure_Movement' in the merged dataset
        if 'Departures_Count' in self.merged_df.columns:
            self.merged_df.rename(columns={'Departures_Count': 'Airport_Daily_Departure_Movement'}, inplace=True)

        # Drop unnecessary columns after merging
        self.merged_df.drop(columns=['datetime_local', 'city', 'Date', 'City'], inplace=True)
        logging.info(f"Merged dataset contains {len(self.merged_df)} rows.")

    def save_merged_data(self, output_path):
        """Save the merged flight, weather, and movement data to a CSV file."""
        self.merged_df.sort_values(by='Departure_Estimated_Local_Time_Out_Gate', inplace=True)
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.merged_df.to_csv(output_path, index=False)
            logging.info(f"Merged data saved to {output_path}")
        except Exception as e:
            logging.error(f"An error occurred while saving the merged data: {e}")
            sys.exit(1)

def main():
    # Configurable parameters
    flight_filepath = '/Users/anis.larid/Desktop/Flight_Delay_Project/1_Data_collection_and_enrichment/Flight_data/flight_data.csv'
    weather_filepath = '/Users/anis.larid/Desktop/Flight_Delay_Project/1_Data_collection_and_enrichment/Weather_data/weather_data.csv'
    movement_filepath = '/Users/anis.larid/Desktop/Flight_Delay_Project/1_Data_collection_and_enrichment/Flight_data/Airport_movement_data/airport_movement_data.csv'
    merged_output_path = '/Users/anis.larid/Desktop/Flight_Delay_Project/1_Data_collection_and_enrichment/LX_flight_delay_data_2024.csv'

    # Initialize the FlightDataProcessor
    processor = FlightDataProcessor(flight_filepath, weather_filepath, movement_filepath)

    # Load, clean, and save the flight, weather, and movement data
    processor.load_data()
    processor.clean_flight_data()
    processor.clean_weather_data()
    processor.clean_movement_data()

    # Merge flight, weather, and movement data, and save the result
    processor.merge_data()
    processor.save_merged_data(merged_output_path)

if __name__ == "__main__":
    main()