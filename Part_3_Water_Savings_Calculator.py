# AGROVOLTAIC SIMULATION - PART 3: WATER SAVINGS CALCULATOR
# =============================================================
# This script reads the shading results from Part 2, calculates the annual
# evapotranspiration (ET) for both an open field and the agrivoltaic system,
# and determines the total annual water savings.

# --- Step 0: Import necessary libraries ---
import pandas as pd
import pyet
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# --- Step 1: Define Inputs ---
# 1a. Input File from Part 2
shading_data_file = 'shading_results_5.0m_pitch.csv'

# 1b. Geographic constants (needed for pyet)
latitude = 41.9028  # Rome, Italy
altitude = 25

print(f"Starting Part 3: Calculating annual water savings.")

# --- Step 2: Load Shading and Weather Data ---
print(f"Loading shading and weather data from '{shading_data_file}'...")
try:
    # Load the data, using the first column as the index.
    df = pd.read_csv(shading_data_file, index_col=0)
    
    # --- FINAL GUARANTEED CORRECTION: Robustly convert the index to a DatetimeIndex ---
    # This method reads the index as text, converts it to a proper UTC datetime,
    # and then localizes it to the correct timezone. This avoids all previous errors.
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Rome')
    
    print("Data loaded successfully.")

except FileNotFoundError:
    print(f"!!!!!!!!!! ERROR !!!!!!!!!!!")
    print(f"Could not find the file '{shading_data_file}'.")
    print("Please make sure you have run Part 2 successfully and the output file is in the same folder.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# --- Step 3: Prepare Data for Daily ET Calculation ---
print("Preparing data for daily evapotranspiration calculation...")
# pyet requires daily data, so we need to resample our hourly data.

# Convert hourly radiation (W/m2) to daily total energy (MJ/m2/day)
# Conversion: (W/m^2 * 3600 s/hr) / 1,000,000 J/MJ = MJ/m^2/hr
# We perform the conversion first, then resample the result.
ghi_mj_per_hour = df['GHI (W/m2)'] * 3600 / 1_000_000
agri_ghi_mj_per_hour = df['Avg GHI Agrivoltaic (W/m2)'] * 3600 / 1_000_000

# Resample all weather variables to get daily min, max, and mean values
daily_df = df.resample('D').agg({
    'Temperature (C)': ['min', 'max', 'mean'],
    'Wind Speed (m/s)': 'mean',
    'Relative Humidity (%)': 'mean'
})

# Flatten the multi-level column names
daily_df.columns = ['tmin', 'tmax', 'tmean', 'wind', 'rh']

# Now, resample and add the daily radiation sums to the daily_df DataFrame
daily_df['sol_rad_open'] = ghi_mj_per_hour.resample('D').sum()
daily_df['sol_rad_agri'] = agri_ghi_mj_per_hour.resample('D').sum()


print("Data preparation complete.")

# --- Step 4: Calculate Annual Evapotranspiration (ET) for Both Scenarios ---
print("Calculating annual ET for both scenarios (this may take a moment)...")

# Scenario A: Open Field (Full Sun)
et_open_field = pyet.pm(
    tmean=daily_df['tmean'],
    wind=daily_df['wind'],
    rs=daily_df['sol_rad_open'],
    elevation=altitude,
    lat=latitude,
    tmax=daily_df['tmax'],
    tmin=daily_df['tmin'],
    rh=daily_df['rh']
)

# Scenario B: Agrivoltaic System (Reduced Sun)
et_agrivoltaic = pyet.pm(
    tmean=daily_df['tmean'],
    wind=daily_df['wind'],
    rs=daily_df['sol_rad_agri'], # Using the reduced radiation under the panels
    elevation=altitude,
    lat=latitude,
    tmax=daily_df['tmax'],
    tmin=daily_df['tmin'],
    rh=daily_df['rh']
)

print("ET calculation complete.")

# --- Step 5: Calculate and Display Final Water Savings ---
# Sum the daily ET values to get the annual total
total_et_open_field = et_open_field.sum()
total_et_agrivoltaic = et_agrivoltaic.sum()

# Calculate the savings in mm and as a percentage
water_savings_mm = total_et_open_field - total_et_agrivoltaic
water_savings_percent = (water_savings_mm / total_et_open_field) * 100

print("\n--- ANNUAL WATER SAVINGS RESULTS ---")
print(f"Total Annual ET in Open Field: {total_et_open_field:.2f} mm")
print(f"Total Annual ET in Agrivoltaic System: {total_et_agrivoltaic:.2f} mm")
print("--------------------------------------")
print(f"Total Annual Water Savings: {water_savings_mm:.2f} mm")
print(f"Percentage Water Savings: {water_savings_percent:.2f}%")
print("--------------------------------------\n")

# --- Step 6: Generate a Plot of Cumulative Water Savings ---
print("Generating a plot of cumulative water savings...")

# Calculate the daily difference in ET
daily_savings = et_open_field - et_agrivoltaic
# Calculate the cumulative sum over the year
cumulative_savings = daily_savings.cumsum()

fig, ax = plt.subplots(figsize=(14, 8))
cumulative_savings.plot(ax=ax, label='Cumulative Water Savings', color='dodgerblue', lw=2.5)

ax.set_title('Cumulative Annual Water Savings (Agrivoltaic vs. Open Field)')
ax.set_ylabel('Total Water Saved (mm)')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Part 3 finished.")
