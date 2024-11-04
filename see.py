import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from scipy.io import loadmat
import streamlit as st

def shift_data(X, y, best_lag):
    if best_lag > 0:
        X_shifted = X[best_lag:]  # Remove first 'best_lag' entries
        y_shifted = y[:-best_lag]  # Remove last 'best_lag' entries
    else:
        X_shifted = X[:best_lag]  # Remove last 'best_lag' entries
        y_shifted = y[-best_lag:]  # Remove first 'best_lag' entries
        
    return X_shifted, y_shifted

# Recursively find the "Tests" folder
def find_tests_folder(base_path):
    for root, dirs, files in os.walk(base_path):
        if "Tests" in dirs:
            return os.path.join(root, "Tests")
    return None

# Base path to the extracted data
base_path = './'

# Check if the "swell_data" folder already exists
if not os.path.exists('swell_data'):
    st.write("Retreiving SWELL data from online......")
    subprocess.run(["wget", "-O", "swell_data.zip", "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n34wcksmts-2.zip"])
    subprocess.run(["unzip", "swell_data.zip", "-d", base_path])
else:
  st.write("Great, SWELL data already present......")

tests_path = find_tests_folder(base_path)

if tests_path:
    st.write(f"'Tests' folder found at: {tests_path}")

    # List of folder names L0 to L8
    folders = [f"L{i}" for i in range(9)]  # L0 to L8

    # Dictionary to store NumPy arrays indexed by folder and file names
    swell_data_dict = {}

    # Initialize the progress bar
    progress_bar = st.progress(0)
    total_files = sum(len(os.listdir(os.path.join(tests_path, folder))) for folder in folders if os.path.exists(os.path.join(tests_path, folder)))

    for folder in folders:
        folder_path = os.path.join(tests_path, folder)
        if os.path.exists(folder_path):
            file_names = [file for file in os.listdir(folder_path) if file.endswith('.mat')]
            total_files_in_folder = len(file_names)

            for i, file_name in enumerate(file_names):
                file_path = os.path.join(folder_path, file_name)

                # Check if file exists to avoid errors
                if os.path.exists(file_path):
                    #st.write("Loading file:", file_path)

                    # Load .mat file
                    mat_data = loadmat(file_path)

                    # Convert mat_data to NumPy arrays
                    numpy_data = {}
                    for key, value in mat_data.items():
                        if isinstance(value, np.ndarray):
                            numpy_data[key] = value
                        elif isinstance(value, list):
                            numpy_data[key] = np.array(value)

                    # Store the NumPy arrays in the data dictionary
                    swell_data_dict[(folder, file_name)] = numpy_data

                    # Update the progress bar
                    progress = (i + 1) / total_files_in_folder
                    progress_bar.progress(progress)
                else:
                    st.write(f"File not found: {file_path}")

    # Display the keys of the dictionary to see the loaded data
    #st.write("Loaded data keys:", swell_data_dict.keys())

else:
    st.write("'Tests' folder not found.")


def extract_data(swell_data_dict, layout, file_name, wave_probe_indices, wave_probe="WP15"):
    wave_elevation = swell_data_dict[(layout, file_name)]["waveElevation_UD"]
    torque = swell_data_dict[(layout, file_name)]["excitationTorque_WE"]

    all_wave_elevations = []
    all_excitation_torques = []

    # Mapping from wave probe string to indices
    wave_probe_mapping = {
        "WP15": [wave_probe_indices[0]],
        #"WP17": [wave_probe_indices[1]],
    }

    # Determine selected probes based on string argument
    if wave_probe is not None:
        selected_wave_probes = wave_probe_mapping.get(wave_probe, [])
    else:
        return None

    for wp_index, torque_index in selected_wave_probes:
        wave_elevation_wp = wave_elevation[wp_index, :].reshape(-1, 1)
        torque_wp = torque[torque_index, :].reshape(-1, 1)
        all_wave_elevations.append(wave_elevation_wp)
        all_excitation_torques.append(torque_wp)

    # Concatenate wave elevations and torques into 2D arrays
    X = np.concatenate(all_wave_elevations, axis=1)
    y = np.concatenate(all_excitation_torques, axis=1)
    st.write(f'Using layout {layout}, wave probe {selected_wave_probes}, and indices: {wave_probe_indices} to extract data')
    return X, y

# Visualize the data with twin axes and a slider for x-axis control
def visualize_data_with_slider(X, y):
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(X, columns=[f'Wave Elevation {i+1}' for i in range(X.shape[1])])
    df['Excitation Torque'] = y

    # Create a plot with twin axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot wave elevations
    for col in df.columns[:-1]:  # Exclude the last column which is 'Excitation Torque'
        ax1.plot(df.index, df[col], label=col)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Wave Elevation', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Wave Elevation and Excitation Torque')
    ax1.grid()

    # Create a second y-axis for the excitation torque
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Excitation Torque'], color='orange', linewidth=2, linestyle='--', label='Excitation Torque')
    ax2.set_ylabel('Excitation Torque', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')


    # Slider for x-axis limits
    lower_limit = st.slider('Lower Limit', 0, len(df) - 50, 0)
    upper_limit = st.slider('Upper Limit', 50, len(df), len(df))

    # Update plot with selected x-axis limits
    ax1.set_xlim(lower_limit, upper_limit)
    ax2.set_xlim(lower_limit, upper_limit)

    # Show the plot
    st.pyplot(fig)

    # Display descriptive statistics
    st.write("Descriptive Statistics:")
    st.write(df.describe())

# Assuming the dictionary swell_data_dict and file names are already defined
wp_15_wave_probe_indices_dict = {
    'L0': [(14, 0)],
    'L1': [(14, 0), (15, 1)],
    'L2': [(14, 0), (16, 1)],
    'L3': [(14, 0), (17, 1)], 
    'L4': [(14, 0), (16, 2)],  
    'L5': [(14, 0), (16, 1)],  
    'L6': [(14, 0), (16, 2)],  
    'L7': [(14, 0), (16, 2)],
}
# Function to visualize the original data
def visualize_original_data(X, y, time_range):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot original X data (within the selected time range)
    ax1.plot(X[time_range[0]:time_range[1]], label='Original X', color='blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X Value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create twin axes for y
    ax2 = ax1.twinx()
    ax2.plot(y[time_range[0]:time_range[1]], label='Original y', color='orange')
    ax2.set_ylabel('y Value', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Original Data Visualization')
    st.pyplot(fig)

# Function to visualize the shifted data
def visualize_shifted_data(X, y, lag, time_range):
    # Shift the data using the selected lag
    X_shifted, y_shifted = shift_data(X, y, lag)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot shifted X data (within the selected time range)
    ax1.plot(X_shifted[time_range[0]:time_range[1]], label='Shifted X', linestyle='--', color='green')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Shifted X Value', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    # Create twin axes for shifted y
    ax2 = ax1.twinx()
    ax2.plot(y_shifted[time_range[0]:time_range[1]], label='y', color='orange')
    ax2.set_ylabel('y Value', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f'Shifted Data Visualization with Lag = {lag}')
    st.pyplot(fig)

# Streamlit interface for selecting layout and file name
data_keys = swell_data_dict.keys()
layouts = sorted(set(key[0] for key in data_keys))
file_names_by_layout = {layout: sorted([key[1] for key in data_keys if key[0] == layout]) for layout in layouts}

layout_select = st.selectbox("Select Layout", layouts, index=2)
file_names = file_names_by_layout[layout_select]
file_name_select = st.selectbox("Select File Name", file_names, index=0)

# Extract data for training
X_train, y_train = extract_data(swell_data_dict, layout_select, file_name_select, wp_15_wave_probe_indices_dict[layout_select])

# Streamlit interface for selecting lag
start_lag = 0
deviation = 100
lag_values = list(range(start_lag - deviation, start_lag + deviation + 1))
selected_lag = st.selectbox("Select Lag", lag_values, index=110)

# Streamlit slider for selecting time range
max_time = min(len(X_train), len(y_train))  # Ensure the time range is within bounds
time_range = st.slider("Select Time Range", 0, max_time, (0, 1000))  # Default range from 0 to 100

# Check if X_train and y_train are defined before plotting
if X_train is not None and y_train is not None:
    visualize_original_data(X_train, y_train, time_range)
    visualize_shifted_data(X_train, y_train, selected_lag, time_range)