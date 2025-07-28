# AGROVOLTAIC SIMULATION - PART 2: PV SYSTEM & SHADING SIMULATOR
# =================================================================
# This script reads the environment data from Part 1, adds a single-axis
# tracking PV system, and calculates the resulting hourly shadow and
# average ground-level irradiance for the entire year.

# --- Step 0: Import necessary libraries ---
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# --- Step 1: Define Inputs ---
# 1a. Input File
environment_data_file = 'environment_report_TMY_real_data.csv'

# 1b. PV System Design (from diagram and user confirmation)
panel_width = 2.38
pivot_height = 2.90
max_tilt_angle = 55
axis_azimuth = 180 # North-South axis for East-West tracking
row_spacing = 6.0  # We will test a single pitch of 6.0m as per the plan

print(f"Starting Part 2: Simulating PV system with a {row_spacing}m pitch.")

# --- Step 2: Load Environment Data ---
print(f"Loading environment data from '{environment_data_file}'...")
try:
    # Load the data, using the first column as the index.
    df = pd.read_csv(environment_data_file, index_col=0)
    
    # --- FINAL GUARANTEED CORRECTION: Robustly convert the index to a DatetimeIndex ---
    # This method reads the index as text, converts it to a proper UTC datetime,
    # and then localizes it to the correct timezone. This avoids all previous errors.
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Rome')
    
    print("Environment data loaded successfully.")
    # Optional: You can uncomment the line below to verify the index type
    # print("Index type after conversion:", type(df.index))

except FileNotFoundError:
    print(f"!!!!!!!!!! ERROR !!!!!!!!!!!")
    print(f"Could not find the file '{environment_data_file}'.")
    print("Please make sure the CSV file from Part 1 is in the same folder as this script.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# --- Step 3: Calculate Single-Axis Tracking with Backtracking ---
print("Calculating hourly panel tilt with backtracking to prevent self-shading...")
# The Ground Coverage Ratio (GCR) is key for backtracking. It's the ratio of panel width to row spacing.
gcr = panel_width / row_spacing

# This function calculates the ideal tilt angle for each hour to follow the sun,
# while adjusting the angle to avoid casting a shadow on the next row.
tracking_data = pvlib.tracking.singleaxis(
    apparent_zenith=df['Sun Elevation (deg)'].apply(lambda x: 90 - x), # pvlib uses zenith (90 - elevation)
    apparent_azimuth=df['Sun Azimuth (deg)'],
    axis_azimuth=axis_azimuth,
    max_angle=max_tilt_angle,
    backtrack=True,
    gcr=gcr
)
# We add the calculated panel tilt to our main dataframe
df['Panel Tilt (deg)'] = tracking_data['surface_tilt']
print("Panel tilt calculation complete.")

# --- Step 4: Simulate Hourly Shading ---
print("Simulating ground shading for every hour of the year...")
# This is a computationally intensive step.

# Define the ground area to analyze, from the start of one row to the next
ground_x = np.arange(0, row_spacing, 0.1) # 10 cm resolution
pivot_x = row_spacing / 2 # The pivot post is in the middle

# Create an empty dataframe to store the shade map (1 for shade, 0 for sun)
is_shaded = pd.DataFrame(0.0, index=df.index, columns=ground_x)

# Loop through every daylight hour of the year
for hour in df[df['Sun Elevation (deg)'] > 0].index:
    # Get the current conditions
    tilt = np.radians(df.loc[hour, 'Panel Tilt (deg)'])
    sun_ele = np.radians(df.loc[hour, 'Sun Elevation (deg)'])
    sun_azi = np.radians(df.loc[hour, 'Sun Azimuth (deg)'])
    
    # Calculate the 3D coordinates of the panel's two long edges
    x1 = pivot_x - 0.5 * panel_width * np.cos(tilt)
    y1 = pivot_height + 0.5 * panel_width * np.sin(tilt)
    x2 = pivot_x + 0.5 * panel_width * np.cos(tilt)
    y2 = pivot_height - 0.5 * panel_width * np.sin(tilt)
    
    # Project the shadow from these two edges onto the ground (the x-axis)
    # The shadow's position depends on the sun's angle in the sky (azimuth)
    shadow_x1 = x1 - y1 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180))
    shadow_x2 = x2 - y2 / np.tan(sun_ele + 1e-6) * np.sin(sun_azi - np.radians(axis_azimuth-180))
    
    # Mark the ground points that fall between the shadow boundaries
    start, end = min(shadow_x1, shadow_x2), max(shadow_x1, shadow_x2)
    is_shaded.loc[hour, (ground_x >= start) & (ground_x <= end)] = 1.0

print("Shading simulation complete.")

# --- Step 5: Calculate Average Ground Irradiance ---
print("Calculating average ground-level irradiance...")
# Convert GHI and DHI to numpy arrays for faster calculation
ghi = df['GHI (W/m2)'].to_numpy()[:, np.newaxis]
dhi = df['DHI (W/m2)'].to_numpy()[:, np.newaxis]

# The irradiance at any point is DHI if shaded, and GHI if not.
# GHI = DNI*cos(zenith) + DHI. So, if a point is shaded, it loses the DNI component.
ground_irradiance = dhi + (ghi - dhi) * (1 - is_shaded)

# Calculate the average across the entire ground surface for each hour
df['Avg GHI Agrivoltaic (W/m2)'] = ground_irradiance.mean(axis=1)
print("Calculation complete.")

# --- Step 6: Save Results and Generate a Verification Plot ---
# Save the key results to a new CSV file for Part 3
output_filename = f"shading_results_{row_spacing}m_pitch.csv"
df[['GHI (W/m2)', 'Avg GHI Agrivoltaic (W/m2)', 'Temperature (C)', 'Wind Speed (m/s)', 'Relative Humidity (%)']].to_csv(output_filename)
print(f"Part 2 finished. Results saved to '{output_filename}'.")

# Generate a plot for a single summer day to verify the results
print("Generating a verification plot for a summer day (June 21st)...")
june_21_data = df[df.index.strftime('%m-%d') == '06-21']

fig, ax = plt.subplots(figsize=(14, 8))
june_21_data['GHI (W/m2)'].plot(ax=ax, label='Open Field (Full Sun)', color='orange', marker='.')
june_21_data['Avg GHI Agrivoltaic (W/m2)'].plot(ax=ax, label=f'Agrivoltaic (Avg at {row_spacing}m pitch)', color='brown', marker='.')

ax.set_title(f'Hourly Solar Radiation on June 21st (Pitch = {row_spacing}m)')
ax.set_ylabel('Irradiance (W/m²)')
ax.set_xlabel('Time of Day')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Verification plot generated.")
