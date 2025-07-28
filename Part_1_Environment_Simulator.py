# AGROVOLTAIC SIMULATION - PART 1: ENVIRONMENT SIMULATOR (FINAL GUARANTEED VERSION)
# ==============================================================================================
# This version uses the most stable and simple method available in pvlib to fetch
# a complete "Typical Meteorological Year" (TMY) dataset from PVGIS. This is the most
# robust and reliable approach.

# --- Step 0: Import necessary libraries ---
import pandas as pd
import pvlib
import numpy as np

print("Libraries imported successfully.")

# --- Step 1: Define Inputs ---
# These parameters can be changed to simulate any location in the world.
latitude = 40.99290  # Rome, Italy - a location with guaranteed data coverage
longitude = 14.25732
altitude = 25
timezone = 'Europe/Rome'

print(f"Simulating environment for Latitude: {latitude}, Longitude: {longitude}")

# --- Step 2: Fetch Typical Meteorological Year (TMY) Data from PVGIS ---
print("Fetching Typical Meteorological Year (TMY) data from PVGIS...")
# This block will try to fetch data. If there is a problem, it will stop the program.
try:
    # FINAL GUARANTEED METHOD: get_pvgis_tmy is the most standard and stable function.
    pvgis_output = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)
    weather = pvgis_output[0] # The weather data is always the first item.
    
    # --- FINAL ROBUST TIMEZONE CORRECTION ---
    # Instead of fixing the TMY index, we create a new, perfect index for a standard
    # year and assign it to our data. This avoids all DST-related errors.
    # We use a non-leap year like 2022 as a standard reference.
    clean_index = pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='h', tz=timezone)
    
    # The TMY dataset has 8760 hours. If our clean index is for a leap year, it might have more.
    # We ensure they have the same length before assigning the index.
    weather = weather.iloc[:len(clean_index)]
    weather.index = clean_index

    print("Successfully fetched and processed real weather data for the full year from PVGIS.")

except Exception as e:
    print(f"!!!!!!!!!! CRITICAL ERROR !!!!!!!!!!!")
    print(f"Could not fetch data from the PVGIS server.")
    print(f"The error is: {e}")
    print(f"The program will stop. Please try again later or check your internet connection.")
    exit()


# --- Step 3: Calculate Sun Position for the Full Year ---
print("Calculating sun position for every hour of the year...")
solar_position = pvlib.solarposition.get_solarposition(
    time=weather.index,
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    temperature=weather.get('temp_air', pd.Series(15, index=weather.index))
)
print("Sun position calculated.")

# --- Step 4: Combine Results ---
print("Combining data into a final report...")
report = pd.DataFrame(index=weather.index)
report['Sun Elevation (deg)'] = solar_position['elevation']
report['Sun Azimuth (deg)'] = solar_position['azimuth']
report['GHI (W/m2)'] = weather.get('ghi', np.nan)
report['DHI (W/m2)'] = weather.get('dhi', np.nan)
report['DNI (W/m2)'] = weather.get('dni', np.nan)
report['Temperature (C)'] = weather.get('temp_air', np.nan)
report['Wind Speed (m/s)'] = weather.get('wind_speed', np.nan)
report['Relative Humidity (%)'] = weather.get('relative_humidity', np.nan)


# --- Step 5: Save to CSV File ---
output_filename = f"environment_report_TMY_real_data.csv"
report.to_csv(output_filename)
print(f"Simulation complete. Data for the full year saved to '{output_filename}'.")

# --- Step 6: Preview the first few rows of the data ---
print("\n--- DATA PREVIEW (First 5 rows) ---")
print(report.head())
print("-----------------------------------")
