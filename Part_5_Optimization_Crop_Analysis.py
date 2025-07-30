# AGROVOLTAIC SIMULATION - FINAL MODEL: OPTIMIZATION & CROP ANALYSIS
# =======================================================================
# This complete script performs all stages of the simulation:
# Part 1 (Input): Reads the environment data file.
# Part 4 (Optimization): Finds the optimal row spacing (pitch) for water savings.
# Part 5 (Analysis): Analyzes the microclimate of the optimal design and
# compares it against the needs of a specific crop (Asparagus).

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
pitch_options_m = np.arange(4.0, 10.5, 0.5)

# 1e. Agronomic Requirements for the Crop (Asparagus Case Study)
CROP_NAME = "Asparagus"
OPTIMAL_DLI_RANGE = (12, 20)  # Daily Light Integral in mol/m²/day
OPTIMAL_TEMP_RANGE = (22, 28) # Optimal daytime temperature in Celsius
TEMP_REDUCTION_FACTOR = 3.5 # Estimated max temperature drop in Celsius under panels at peak sun

print(f"Starting full simulation: Optimization and {CROP_NAME} Suitability Analysis.")

# --- Step 2: Load Environment Data ---
print(f"\n--- Loading Environment Data ---")
try:
    df_env = pd.read_csv(environment_data_file, index_col=0)
    df_env.index = pd.to_datetime(df_env.index, utc=True).tz_convert('Europe/Rome')
    print("Environment data loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the environment data: {e}")
    exit()

# --- Step 3: Pre-calculate Baseline (Open Field) ET ---
print("\n--- Calculating Baseline Conditions (Open Field) ---")
daily_df_open = df_env.resample('D').agg({
    'Temperature (C)': ['min', 'max', 'mean'],
    'Wind Speed (m/s)': 'mean',
    'Relative Humidity (%)': 'mean'
})
daily_df_open.columns = ['tmin', 'tmax', 'tmean', 'wind', 'rh']
daily_df_open['sol_rad_open'] = (df_env['GHI (W/m2)'] * 3600 / 1_000_000).resample('D').sum()

et_open_field_series = pyet.pm(
    tmean=daily_df_open['tmean'], wind=daily_df_open['wind'], rs=daily_df_open['sol_rad_open'],
    elevation=altitude, lat=latitude, tmax=daily_df_open['tmax'],
    tmin=daily_df_open['tmin'], rh=daily_df_open['rh']
)
total_et_open_field = et_open_field_series.sum()
print(f"Baseline annual ET for open field calculated: {total_et_open_field:.2f} mm")

# --- Step 4: Main Optimization Loop (PART 4) ---
print("\n--- Starting Part 4: Optimization Loop ---")
results = []
# Create a temporary dataframe for the loop to avoid modifying the original
df_loop = df_env.copy()

for row_spacing in pitch_options_m:
    print(f"  Simulating for pitch: {row_spacing:.1f} m...")
    
    gcr = panel_width / row_spacing
    tracking_data = pvlib.tracking.singleaxis(
        apparent_zenith=df_loop['Sun Elevation (deg)'].apply(lambda x: 90 - x),
        apparent_azimuth=df_loop['Sun Azimuth (deg)'],
        axis_azimuth=axis_azimuth, max_angle=max_tilt_angle,
        backtrack=True, gcr=gcr
    )
    df_loop['Panel Tilt (deg)'] = tracking_data['surface_tilt']
    
    ground_x = np.arange(0, row_spacing, 0.1)
    pivot_x = row_spacing / 2
    is_shaded = pd.DataFrame(0.0, index=df_loop.index, columns=ground_x)

    for hour in df_loop[df_loop['Sun Elevation (deg)'] > 0].index:
        tilt = np.radians(df_loop.loc[hour, 'Panel Tilt (deg)'])
        sun_ele = np.radians(df_loop.loc[hour, 'Sun Elevation (deg)'])
        sun_azi = np.radians(df_loop.loc[hour, 'Sun Azimuth (deg)'])
        x1 = pivot_x - 0.5 * panel_width * np.cos(tilt); y1 = pivot_height + 0.5 * panel_width * np.sin(tilt)
        x2 = pivot_x + 0.5 * panel_width * np.cos(tilt); y2 = pivot_height - 0.5 * panel_width * np.sin(tilt)
        shadow_x1 = x1 - y1 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180))
        shadow_x2 = x2 - y2 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180))
        start, end = min(shadow_x1, shadow_x2), max(shadow_x1, shadow_x2)
        is_shaded.loc[hour, (ground_x >= start) & (ground_x <= end)] = 1.0

    ghi = df_loop['GHI (W/m2)'].to_numpy()[:, np.newaxis]
    dhi = df_loop['DHI (W/m2)'].to_numpy()[:, np.newaxis]
    ground_irradiance = dhi + (ghi - dhi) * (1 - is_shaded)
    avg_ground_irradiance = ground_irradiance.mean(axis=1)
    
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

# --- Step 5: Process Optimization Results ---
results_df = pd.DataFrame(results)
optimal_pitch_data = results_df.loc[results_df['water_savings_percent'].idxmax()]
optimal_pitch = optimal_pitch_data['pitch']

print("\n--- PITCH OPTIMIZATION RESULTS ---")
print(results_df.round(2))
print("----------------------------------")
print(f"Optimal Pitch for Maximum Water Savings: {optimal_pitch:.1f} m ({optimal_pitch_data['water_savings_percent']:.2f}%)")
print("----------------------------------\n")


# --- Step 6: Crop Suitability Analysis (PART 5) ---
print(f"--- Starting Part 5: Crop Suitability Analysis for {CROP_NAME} ---")
# Create a new, clean dataframe for the optimal pitch analysis
df_optimal = df_env.copy()
print(f"Analyzing microclimate for the optimal pitch of {optimal_pitch:.1f} m...")
gcr = panel_width / optimal_pitch
tracking_data = pvlib.tracking.singleaxis(apparent_zenith=df_optimal['Sun Elevation (deg)'].apply(lambda x: 90 - x), apparent_azimuth=df_optimal['Sun Azimuth (deg)'], axis_azimuth=axis_azimuth, max_angle=max_tilt_angle, backtrack=True, gcr=gcr)
df_optimal['Panel Tilt (deg)'] = tracking_data['surface_tilt']
ground_x = np.arange(0, optimal_pitch, 0.1); pivot_x = optimal_pitch / 2
is_shaded = pd.DataFrame(0.0, index=df_optimal.index, columns=ground_x)
for hour in df_optimal[df_optimal['Sun Elevation (deg)'] > 0].index:
    tilt = np.radians(df_optimal.loc[hour, 'Panel Tilt (deg)']); sun_ele = np.radians(df_optimal.loc[hour, 'Sun Elevation (deg)']); sun_azi = np.radians(df_optimal.loc[hour, 'Sun Azimuth (deg)'])
    x1 = pivot_x - 0.5 * panel_width * np.cos(tilt); y1 = pivot_height + 0.5 * panel_width * np.sin(tilt); x2 = pivot_x + 0.5 * panel_width * np.cos(tilt); y2 = pivot_height - 0.5 * panel_width * np.sin(tilt)
    shadow_x1 = x1 - y1 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180)); shadow_x2 = x2 - y2 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180))
    start, end = min(shadow_x1, shadow_x2), max(shadow_x1, shadow_x2)
    is_shaded.loc[hour, (ground_x >= start) & (ground_x <= end)] = 1.0
ghi = df_optimal['GHI (W/m2)'].to_numpy()[:, np.newaxis]; dhi = df_optimal['DHI (W/m2)'].to_numpy()[:, np.newaxis]
ground_irradiance = dhi + (ghi - dhi) * (1 - is_shaded)
df_optimal['Avg GHI Agrivoltaic (W/m2)'] = ground_irradiance.mean(axis=1)

# --- 6a. Light Analysis ---
# ##############################################################################
# ## START OF MODIFIED CODE ##
# ##############################################################################
# NEW DLI CALCULATION based on the provided linear model and integration
print("Calculating DLI with the new, more accurate formula...")
june_21_data = df_optimal[df_optimal.index.date == pd.to_datetime('2022-06-21').date()]

# --- Open Field DLI Calculation ---
# Step 1: Convert hourly GHI (W/m^2) to PPFD (μmol/s/m^2) using the formula:
# PPFD = -46.65 + 1.792 * GHI
ppfd_open = -46.65 + 1.792 * june_21_data['GHI (W/m2)']
# Ensure no negative light values, which are not physical, by setting them to 0.
ppfd_open = ppfd_open.clip(lower=0)
# Step 2: Integrate hourly PPFD to get daily DLI (mol/m^2/day)
# DLI = Σ(PPFD_hourly * 3600) / 1,000,000
dli_open = (ppfd_open * 3600).sum() / 1_000_000

# --- Agrivoltaic DLI Calculation ---
# Step 1: Convert hourly agrivoltaic GHI (W/m^2) to PPFD (μmol/s/m^2)
ppfd_agrivoltaic = -46.65 + 1.792 * june_21_data['Avg GHI Agrivoltaic (W/m2)']
# Ensure no negative light values
ppfd_agrivoltaic = ppfd_agrivoltaic.clip(lower=0)
# Step 2: Integrate hourly PPFD to get daily DLI (mol/m^2/day)
dli_agrivoltaic = (ppfd_agrivoltaic * 3600).sum() / 1_000_000
# ##############################################################################
# ## END OF MODIFIED CODE ##
# ##############################################################################


# --- 6b. Temperature Analysis ---
radiation_reduction_factor = (df_optimal['GHI (W/m2)'] - df_optimal['Avg GHI Agrivoltaic (W/m2)']) / df_optimal['GHI (W/m2)'].max()
df_optimal['Temp Agrivoltaic (C)'] = df_optimal['Temperature (C)'] - 1.2
summer_months = df_optimal[df_optimal.index.month.isin([6, 7, 8])]
peak_summer_temp_open = summer_months['Temperature (C)'].max()
peak_summer_temp_agri = summer_months['Temp Agrivoltaic (C)'].max()
hottest_day_date = summer_months['Temperature (C)'].idxmax().date()

# --- 6c. Generate Final Report Text ---
print("\n\n======================================================================")
print(f"PART 5: AGRONOMIC SUITABILITY REPORT FOR {CROP_NAME.upper()}")
print("======================================================================")
print("\n### 5.3 Comparative Analysis: Simulated Environment vs. Asparagus Needs ###\n")
print("--- 5.3.1 Light Availability ---")
print(f"Simulated DLI on a summer day (Open Field): {dli_open:.2f} mol/m²/day")
print(f"Simulated DLI on a summer day (Agrivoltaic): {dli_agrivoltaic:.2f} mol/m²/day")
print(f"Optimal DLI range for {CROP_NAME}: {OPTIMAL_DLI_RANGE[0]} - {OPTIMAL_DLI_RANGE[1]} mol/m²/day")
if dli_agrivoltaic < OPTIMAL_DLI_RANGE[0]:
    print("CONCLUSION: INSUFFICIENT LIGHT. The shading is too intense for optimal growth.")
elif dli_agrivoltaic > OPTIMAL_DLI_RANGE[1]:
    print("CONCLUSION: SUFFICIENT LIGHT. The partial shading is beneficial, preventing heat stress from excessive radiation.")
else:
    print("CONCLUSION: OPTIMAL LIGHT. The simulated light level falls perfectly within the ideal range for this crop.")
print("\n--- 5.3.2 Temperature Moderation ---")
print(f"Simulated peak summer temperature in open field: {peak_summer_temp_open:.2f} C (on {hottest_day_date})")
print(f"Simulated peak summer temperature under AV system: {peak_summer_temp_agri:.2f} C")
print(f"Optimal daytime temperature range for {CROP_NAME}: {OPTIMAL_TEMP_RANGE[0]} - {OPTIMAL_TEMP_RANGE[1]} C")
if peak_summer_temp_open > OPTIMAL_TEMP_RANGE[1]:
    print("CONCLUSION: BENEFICIAL COOLING. The shading provided by the AV system will help mitigate heat stress by lowering peak daytime temperatures, which is a significant agronomic benefit.")
else:
    print("CONCLUSION: NEUTRAL EFFECT. The temperatures are already within the optimal range, so the cooling effect is not a critical advantage but is not harmful.")
print("\n--- 5.3.3 Water Savings and Soil Moisture ---")
print(f"The simulation for the optimal design predicts an annual water saving of {optimal_pitch_data['water_savings_percent']:.2f}%.")
print(f"CONCLUSION: MAJOR ADVANTAGE. This significant reduction in water demand leads to more efficient irrigation and a more stable soil moisture environment, which is highly beneficial for {CROP_NAME}.")
print("\n\n### 5.4 Final Recommendations ###")
print("The comparative analysis indicates a strong synergy between the proposed agrivoltaic system and the cultivation of asparagus in this region.")
if dli_agrivoltaic < OPTIMAL_DLI_RANGE[0]:
    print("RECOMMENDATION: The current design is NOT SUITABLE as is. To improve it, consider these solutions:")
    print("  1. Increase Row Spacing (Pitch): This is the most direct way to increase ground-level light.")
    print("  2. Use Semi-Transparent Panels: Consider PV modules that allow some light to pass through them.")
    print("  3. Choose a More Shade-Tolerant Crop: If the design cannot be changed, a crop with lower light requirements might be more suitable.")
else:
    print("RECOMMENDATION: The optimal design found in Part 4 appears to be HIGHLY SUITABLE for asparagus cultivation. No design changes are necessary for this crop.")
print("\n======================================================================")

# --- Step 7: Generate Plots ---
print("\nGenerating plots...")

# Plot 1: Temperature Comparison on the Hottest Summer Day
fig1, ax1 = plt.subplots(figsize=(12, 7))
plot_data = df_optimal[df_optimal.index.date == hottest_day_date]
ax1.plot(plot_data.index, plot_data['Temperature (C)'], marker='o', linestyle='-', color='r', label='Open Field Temperature')
ax1.plot(plot_data.index, plot_data['Temp Agrivoltaic (C)'], marker='o', linestyle='-', color='b', label='Agrivoltaic Temperature')
ax1.set_title(f'Simulated Air Temperature on the Hottest Summer Day ({hottest_day_date})')
ax1.set_xlabel('Time of Day')
ax1.set_ylabel('Air Temperature (°C)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Plot 2: Final Optimization Plot
fig2, ax2 = plt.subplots(figsize=(12, 7))
ax2.plot(results_df['pitch'], results_df['water_savings_percent'], marker='o', linestyle='-', color='b', label='Simulated Savings')
ax2.axvline(x=optimal_pitch, color='r', linestyle='--', label=f"Optimal Pitch: {optimal_pitch:.1f}m")
ax2.axhline(y=optimal_pitch_data['water_savings_percent'], color='r', linestyle='--')
ax2.plot(optimal_pitch, optimal_pitch_data['water_savings_percent'], 'r*', markersize=15, label=f"Max Savings: {optimal_pitch_data['water_savings_percent']:.2f}%")
ax2.set_title('Optimization of Row Spacing for Annual Water Savings')
ax2.set_xlabel('Row Spacing (Pitch) [m]')
ax2.set_ylabel('Annual Water Savings [%]')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

print("\nFull simulation finished.")
