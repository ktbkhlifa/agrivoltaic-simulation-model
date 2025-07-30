# AGROVOLTAIC SIMULATION - PART 4: OPTIMIZATION ENGINE
# =======================================================
# This script automates the entire simulation process (Parts 2 and 3)
# across a range of different row spacings (pitches) to find the optimal
# design that maximizes annual water savings.

# --- Step 0: Import necessary libraries ---
import pandas as pd
import numpy as np
import pvlib
import pyet
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# --- Step 1: Define Inputs and Constants ---
# 1a. Input File from Part 1
environment_data_file = 'environment_report_TMY_real_data.csv'

# 1b. PV System Design
panel_width = 2.38
pivot_height = 2.90
max_tilt_angle = 55
axis_azimuth = 180

# 1c. Geographic constants
latitude = 41.9028
altitude = 25

# 1d. Optimization settings
# Define the range of row spacings to test (from 4.0m to 10.0m)
pitch_options_m = np.arange(4.0, 10.5, 0.5)

print(f"Starting Part 4: Optimizing for pitch from {pitch_options_m[0]}m to {pitch_options_m[-1]}m.")

# --- Step 2: Load Environment Data ---
print(f"Loading environment data from '{environment_data_file}'...")
try:
    df_env = pd.read_csv(environment_data_file, index_col=0)
    df_env.index = pd.to_datetime(df_env.index, utc=True).tz_convert('Europe/Rome')
    print("Environment data loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the environment data: {e}")
    exit()

# --- Step 3: Pre-calculate Baseline (Open Field) ET ---
# This only needs to be done once, making the loop much faster.
print("Pre-calculating the baseline annual ET for an open field...")
# Prepare daily data for the open field scenario
daily_df_open = df_env.resample('D').agg({
    'Temperature (C)': ['min', 'max', 'mean'],
    'Wind Speed (m/s)': 'mean',
    'Relative Humidity (%)': 'mean'
})
daily_df_open.columns = ['tmin', 'tmax', 'tmean', 'wind', 'rh']
daily_df_open['sol_rad_open'] = (df_env['GHI (W/m2)'] * 3600 / 1_000_000).resample('D').sum()

# Calculate daily ET for the entire year for the open field
et_open_field_series = pyet.pm(
    tmean=daily_df_open['tmean'], wind=daily_df_open['wind'], rs=daily_df_open['sol_rad_open'],
    elevation=altitude, lat=latitude, tmax=daily_df_open['tmax'],
    tmin=daily_df_open['tmin'], rh=daily_df_open['rh']
)
total_et_open_field = et_open_field_series.sum()
print(f"Baseline annual ET for open field calculated: {total_et_open_field:.2f} mm")


# --- Step 4: Main Optimization Loop ---
print("Starting optimization loop...")
results = []

for row_spacing in pitch_options_m:
    print(f"  Simulating for pitch: {row_spacing:.1f} m...")
    
    # --- Part 2 Logic (Shading Simulation) ---
    gcr = panel_width / row_spacing
    tracking_data = pvlib.tracking.singleaxis(
        apparent_zenith=df_env['Sun Elevation (deg)'].apply(lambda x: 90 - x),
        apparent_azimuth=df_env['Sun Azimuth (deg)'],
        axis_azimuth=axis_azimuth, max_angle=max_tilt_angle,
        backtrack=True, gcr=gcr
    )
    df_env['Panel Tilt (deg)'] = tracking_data['surface_tilt']
    
    ground_x = np.arange(0, row_spacing, 0.1)
    pivot_x = row_spacing / 2
    is_shaded = pd.DataFrame(0.0, index=df_env.index, columns=ground_x)

    for hour in df_env[df_env['Sun Elevation (deg)'] > 0].index:
        tilt = np.radians(df_env.loc[hour, 'Panel Tilt (deg)'])
        sun_ele = np.radians(df_env.loc[hour, 'Sun Elevation (deg)'])
        sun_azi = np.radians(df_env.loc[hour, 'Sun Azimuth (deg)'])
        x1 = pivot_x - 0.5 * panel_width * np.cos(tilt); y1 = pivot_height + 0.5 * panel_width * np.sin(tilt)
        x2 = pivot_x + 0.5 * panel_width * np.cos(tilt); y2 = pivot_height - 0.5 * panel_width * np.sin(tilt)
        shadow_x1 = x1 - y1 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180))
        shadow_x2 = x2 - y2 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180))
        start, end = min(shadow_x1, shadow_x2), max(shadow_x1, shadow_x2)
        is_shaded.loc[hour, (ground_x >= start) & (ground_x <= end)] = 1.0

    ghi = df_env['GHI (W/m2)'].to_numpy()[:, np.newaxis]
    dhi = df_env['DHI (W/m2)'].to_numpy()[:, np.newaxis]
    ground_irradiance = dhi + (ghi - dhi) * (1 - is_shaded)
    avg_ground_irradiance = ground_irradiance.mean(axis=1)

    # --- Part 3 Logic (Water Savings Calculation) ---
    sol_rad_agri_mj_day = (avg_ground_irradiance * 3600 / 1_000_000).resample('D').sum()
    
    et_agrivoltaic_series = pyet.pm(
        tmean=daily_df_open['tmean'], wind=daily_df_open['wind'], rs=sol_rad_agri_mj_day,
        elevation=altitude, lat=latitude, tmax=daily_df_open['tmax'],
        tmin=daily_df_open['tmin'], rh=daily_df_open['rh']
    )
    total_et_agrivoltaic = et_agrivoltaic_series.sum()
    
    water_savings_percent = ((total_et_open_field - total_et_agrivoltaic) / total_et_open_field) * 100
    results.append({'pitch': row_spacing, 'water_savings_percent': water_savings_percent})

print("Optimization loop finished.")

# --- Step 5: Display and Save Final Results ---
results_df = pd.DataFrame(results)
optimal_pitch_data = results_df.loc[results_df['water_savings_percent'].idxmax()]
optimal_pitch = optimal_pitch_data['pitch']

print("\n--- PITCH OPTIMIZATION RESULTS ---")
print(results_df.round(2))
print("----------------------------------")
print(f"Optimal Pitch for Maximum Water Savings: {optimal_pitch:.1f} m ({optimal_pitch_data['water_savings_percent']:.2f}%)")
print("----------------------------------\n")

# NEW: Save the results to a CSV file that can be opened in Excel
print("Saving optimization results to a CSV file...")
output_filename_summary = 'pitch_optimization_summary.csv'
try:
    results_df.round(2).to_csv(output_filename_summary, index=False)
    print(f"Successfully saved results to '{output_filename_summary}'")
except Exception as e:
    print(f"Error saving file: {e}")

# --- Step 6: Generate Final Optimization Plot ---
print("Generating the final optimization plot...")
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(results_df['pitch'], results_df['water_savings_percent'], marker='o', linestyle='-', color='b', label='Simulated Savings')

# Highlight the optimal point
ax.axvline(x=optimal_pitch, color='r', linestyle='--', label=f"Optimal Pitch: {optimal_pitch:.1f}m")
ax.axhline(y=optimal_pitch_data['water_savings_percent'], color='r', linestyle='--')
ax.plot(optimal_pitch, optimal_pitch_data['water_savings_percent'], 'r*', markersize=15, label=f"Max Savings: {optimal_pitch_data['water_savings_percent']:.2f}%")

ax.set_title('Optimization of Row Spacing for Annual Water Savings')
ax.set_xlabel('Row Spacing (Pitch) [m]')
ax.set_ylabel('Annual Water Savings [%]')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Part 4 finished.")
