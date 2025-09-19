import scipy.io as sio
import numpy as np

# Load the .mat file
data = sio.loadmat("_2cf6b51ab50ccd54c3c3676b5c22d0b6_c1p8.mat")

# Extract the stimulus and spike train
stim = data['stim'].flatten()   # stimulus vector
rho = data['rho'].flatten()     # spike train (0s and 1s)

# Define parameters
sampling_rate = 500  # Hz
sampling_period = 1000 / sampling_rate  # in ms → 2 ms
window_ms = 300

# Number of timesteps in 300 ms (same as num_timesteps in MATLAB)
num_timesteps = int(window_ms / sampling_period)  # → 150

# Count spikes after the first 300 ms
num_spikes = np.sum(rho[num_timesteps:])

print(f"Sampling period (ms): {sampling_period}")
print(f"Number of timesteps in {window_ms} ms: {num_timesteps}")
print(f"Number of spikes after first {window_ms} ms: {num_spikes}")