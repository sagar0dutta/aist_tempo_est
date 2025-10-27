import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter, argrelmax


def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')






def compute_tempogram(dir_change_onset, sampling_rate, window_length, hop_size, tempi=np.arange(45, 140, 1)):
    
    # tempogram_raw = []
    # tempogram_ab = []
    for i in range(dir_change_onset.shape[1]):
        
        hann_window = np.hanning(window_length)
        half_window_length = window_length // 2
        signal_length = len(dir_change_onset[:,i])
        left_padding = half_window_length
        right_padding = half_window_length
        padded_signal_length = signal_length + left_padding + right_padding
        
        # Extend the signal with zeros at both ends
        padded_signal = np.concatenate((np.zeros(left_padding), dir_change_onset[:,i], np.zeros(right_padding)))
        time_indices = np.arange(padded_signal_length)
        num_frames = int(np.floor(padded_signal_length - window_length) / hop_size) + 1
        num_tempo_values = len(tempi)
        tempogram = np.zeros((num_tempo_values, num_frames), dtype=np.complex_)
        
        time_axis_seconds = np.arange(num_frames) * hop_size / sampling_rate
        tempo_axis_bpm = tempi
        
        for tempo_idx in range(num_tempo_values):   # frequency axis
            frequency = (tempi[tempo_idx] / 60) / sampling_rate
            complex_exponential = np.exp(-2 * np.pi * 1j * frequency * time_indices)
            modulated_signal = padded_signal * complex_exponential
            
            for frame_idx in range(num_frames): # time axis
                start_index = frame_idx * hop_size
                end_index = start_index + window_length
                tempogram[tempo_idx, frame_idx] = np.sum(hann_window * modulated_signal[start_index:end_index])    
                
        tempogram_raw= tempogram
        tempogram_ab= np.abs(tempogram)
    # print("Tempograms generated")
 
    return tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm



def dance_tempo_estimation_single(tempogram_ab, tempogram_raw, sampling_rate, novelty_length, window_length, hop_size, tempi):
    """Compute windowed sinusoid with optimal phase


    Args:
        tempogram_ab (list): list of arrays of Fourier-based abs tempogram 
        tempogram_raw (list): list of arrays of Fourier-based (complex-valued) tempogram
        sampling_rate (scalar): Sampling rate
        novelty_length (int): Length of novelty curve
        window_length (int): Window length
        hop_size (int): Hop size
        tempi (np.ndarray): Set of tempi (given in BPM)

    Returns:
        beat
    """

    hann_window = np.hanning(window_length)
    half_window_length = window_length // 2
    padded_curve_length = novelty_length + half_window_length
    estimated_beat_pulse = np.zeros(padded_curve_length)

    num_frames = tempogram_raw.shape[1]
    bpm_list, mag_list, phase_list = [], [], []


    for frame_idx in range(num_frames):
        # strongest tempo bin for current frame
        peak_idx = np.argmax(tempogram_ab[:, frame_idx])
        peak_bpm = tempi[peak_idx]
        frequency = (peak_bpm / 60) / sampling_rate  # Hz → samples
        
        complex_value = tempogram_raw[peak_idx, frame_idx]
        phase = -np.angle(complex_value) / (2 * np.pi)
        magnitude = np.abs(complex_value)
        
        # Reconstruct sinusoidal kernel
        start_index = frame_idx * hop_size
        end_index = start_index + window_length
        time_kernel = np.arange(start_index, end_index)
        sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * frequency - phase))

        # Accumulate weighted sinusoid
        valid_end = min(end_index, padded_curve_length)
        valid_len = valid_end - start_index
        if valid_len > 0:
            estimated_beat_pulse[start_index:valid_end] += magnitude * sinusoidal_kernel[:valid_len]

        mag_list.append(magnitude)
        bpm_list.append(peak_bpm)
        phase_list.append(phase)
        

    json_data = {"estimated_beat_pulse": estimated_beat_pulse,
                 "mag_arr": np.array(mag_list),
                 "bpm_arr": np.array(bpm_list),}

    return json_data

############# Multi Segment ################

# --- Helper functions ---
def best_alignment(x, f, fps):
    """Return normalized correlation (−1 to 1) and lag (sec) for x vs sine kernel at freq f."""
    x = np.asarray(x).flatten()
    x = x - np.mean(x)

    # Skip empty/constant signals early
    if np.allclose(x, 0) or np.std(x) < 1e-8:
        return 0.0, 0.0, np.zeros_like(x), np.arange(len(x)) / fps

    # Reference sinusoid
    sine = np.sin(2 * np.pi * f * np.arange(len(x)) / fps)
    sine -= np.mean(sine)

    denom = np.sqrt(np.sum(x**2) * np.sum(sine**2))
    if denom < 1e-12:  # avoid div-by-zero
        return 0.0, 0.0, np.zeros_like(x), np.arange(len(x)) / fps

    corr = np.correlate(x, sine, mode="full") / denom
    lags = np.arange(-len(x) + 1, len(x)) / fps

    best_idx = np.argmax(np.abs(corr))
    best_corr = float(corr[best_idx])
    best_lag = float(lags[best_idx])
    return best_corr, best_lag, corr, lags


# def dance_tempo_estimation_multi(tempogram_ab_list, tempogram_raw_list, anchors, fps, novelty_length, window_length, hop_size, tempi):
#     """Compute windowed sinusoid with optimal phase


#     Args:
#         tempogram_ab (list): list of arrays of Fourier-based abs tempogram 
#         tempogram_raw (list): list of arrays of Fourier-based (complex-valued) tempogram
#         sampling_rate (scalar): Sampling rate
#         novelty_length (int): Length of novelty curve
#         window_length (int): Window length
#         hop_size (int): Hop size
#         tempi (np.ndarray): Set of tempi (given in BPM)

#     Returns:
#         beat
#     """

#     hann_window = np.hanning(window_length)
#     half_window_length = window_length // 2
#     padded_curve_length = novelty_length + half_window_length
#     estimated_beat_pulse = np.zeros(padded_curve_length)

#     # tempogram_data = {}
#     gtempo_list = []
#     for tempogram_ab, tempogram_raw in zip(tempogram_ab_list, tempogram_raw_list):

#         num_frames = tempogram_raw.shape[1]
#         bpm_list,complex_list, mag_list, phase_list = [], [], [], []

#         for frame_idx in range(num_frames):
#             # strongest tempo bin for current frame
#             peak_idx = np.argmax(tempogram_ab[:, frame_idx])
#             peak_bpm = tempi[peak_idx]
#             frequency = (peak_bpm / 60) / fps  # Hz → samples
            
#             complex_value = tempogram_raw[peak_idx, frame_idx]
#             phase = -np.angle(complex_value) / (2 * np.pi)
#             magnitude = np.abs(complex_value)
            
#             # Reconstruct sinusoidal kernel
#             start_index = frame_idx * hop_size
#             end_index = start_index + window_length
#             time_kernel = np.arange(start_index, end_index)
#             sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * frequency - phase))

#             # Accumulate weighted sinusoid
#             valid_end = min(end_index, padded_curve_length)
#             valid_len = valid_end - start_index
#             if valid_len > 0:
#                 estimated_beat_pulse[start_index:valid_end] += magnitude * sinusoidal_kernel[:valid_len]

#             mag_list.append(magnitude)
#             bpm_list.append(peak_bpm)
#             phase_list.append(phase)
#             complex_list.append(complex_value)
            
#         gtempo_list.append(np.median(bpm_list))    
        
#     test_freq = []
#     for gbpm in gtempo_list:
#         fq = np.round(gbpm/60, 2)  # in Hz
#         test_freq.append(fq)


#     anchor_names = ["anchor1", "anchor2", "anchor3"]     # TODO

#     results = {name: {} for name in anchor_names}
#     best_global = {"freq": None, "sensor": None, "corr": 0.0, "lag": None}

#     for f in test_freq:
#         for name, x in zip(anchor_names, anchors):
#             best_corr, best_lag, corr, lags = best_alignment(x, f, fps)
#             results[name][f] = (best_corr, best_lag)

#             if abs(best_corr) > abs(best_global["corr"]):
#                 best_global.update({"freq": f, "sensor": name, "corr": best_corr, "lag": best_lag})
    
#     if best_global['freq'] is None:
#         gtempo = gtempo_list[-1]  # in BPM
#     else:    
#         gtempo = np.round(best_global['freq']*60, 2)  # in BPM
        
#     tempogram_data = {"gtempo": gtempo,}

#     return tempogram_data

def dance_tempo_estimation(tempogram_ab_list, tempogram_raw_list, fps, novelty_length, window_length, hop_size, tempi):
    """Compute windowed sinusoid with optimal phase


    Args:
        tempogram_ab (list): list of arrays of Fourier-based abs tempogram 
        tempogram_raw (list): list of arrays of Fourier-based (complex-valued) tempogram
        sampling_rate (scalar): Sampling rate
        novelty_length (int): Length of novelty curve
        window_length (int): Window length
        hop_size (int): Hop size
        tempi (np.ndarray): Set of tempi (given in BPM)

    Returns:
        beat
    """


    hann_window = np.hanning(window_length)
    half_window_length = window_length // 2
    padded_curve_length = novelty_length + half_window_length
    estimated_beat_pulse = np.zeros(padded_curve_length)

    tempogram_data = []
    # gtempo_list = []
    for tempogram_ab, tempogram_raw in zip(tempogram_ab_list, tempogram_raw_list):

        num_frames = tempogram_raw.shape[1]
        bpm_list,complex_list, mag_list, phase_list = [], [], [], []

        for frame_idx in range(num_frames):
            # strongest tempo bin for current frame
            peak_idx = np.argmax(tempogram_ab[:, frame_idx])
            peak_bpm = tempi[peak_idx]
            frequency = (peak_bpm / 60) / fps  # Hz → samples
            
            complex_value = tempogram_raw[peak_idx, frame_idx]
            phase = -np.angle(complex_value) / (2 * np.pi)
            magnitude = np.abs(complex_value)
            
            # Reconstruct sinusoidal kernel
            start_index = frame_idx * hop_size
            end_index = start_index + window_length
            time_kernel = np.arange(start_index, end_index)
            sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * frequency - phase))

            # Accumulate weighted sinusoid
            valid_end = min(end_index, padded_curve_length)
            valid_len = valid_end - start_index
            if valid_len > 0:
                estimated_beat_pulse[start_index:valid_end] += magnitude * sinusoidal_kernel[:valid_len]

            mag_list.append(magnitude)
            bpm_list.append(peak_bpm)
            phase_list.append(phase)
            complex_list.append(complex_value)
            
            
        # gtempo_list.append(np.median(bpm_list))    
        tempogram_data.append({
            "magnitude": mag_list,
            "bpm": bpm_list,
            "phase": phase_list,
            "complex": complex_list,
            "median_tempo": np.median(bpm_list)
            })
        
        

    return tempogram_data


def filter_dir_onsets_by_threshold(dir_change_array, threshold_s=0.25, fps=60):
    # Removes any onsets that fall within the threshold window after the current onset
    # dir_change_array is of size (n,3) and is a binary array where onset is represented by value >0
    
    window_frames = int(threshold_s * fps)  # Calculate the window size in frames
    filtered_col = []
    
    for col in range(dir_change_array.shape[1]):
        dir_change_onsets = dir_change_array[:, col]
        dir_change_frames = np.where(dir_change_onsets > 0)[0]  # Extract indices of non-zero values
        
        dir_new_onsets = np.zeros(len(dir_change_onsets))  # Initialize the new onsets array
        filtered_onsets = []  # To store the filtered onsets
        
        i = 0
        while i < len(dir_change_frames):
            current_frame_onset = dir_change_frames[i]
            end_frame = current_frame_onset + window_frames
            
            # Add the current onset to the filtered list
            filtered_onsets.append(current_frame_onset)
            
            # Skip all subsequent onsets that fall within the window
            j = i + 1
            while j < len(dir_change_frames) and dir_change_frames[j] <= end_frame:
                j += 1
            
            # Update the index to the next onset that is outside the window
            i = j
        
        # Set filtered onsets in the new onset array
        dir_new_onsets[filtered_onsets] = 1
        filtered_col.append(dir_new_onsets)
    
    # Stack filtered columns to create the final filtered array
    filtered_array = np.column_stack(filtered_col)
    
    return filtered_array


def smooth_velocity(velocity_data, abs='yes', window_length = 60, polyorder = 0):
    # velocity_data consist velocity of 3 axis and its size is (n, 3)
    
    veltemp_list = []
    for i in range(velocity_data.shape[1]):
        smoothed_velocity = savgol_filter(velocity_data[:, i], window_length, polyorder)
        if abs== 'yes':
            smoothed_velocity = np.abs(smoothed_velocity)

        veltemp_list.append(smoothed_velocity)
    smooth_vel_arr = np.column_stack(veltemp_list)  # Stacking the list along axis 1 to make an (n, 3) array
    
    return smooth_vel_arr



##### april 2025
def velocity_based_novelty(velocity_array, height = 0.2, distance=15):
    dir_change_onset_arr = []
    
    
    for i in range(velocity_array.shape[1]):
        # Find peaks (local maxima) with a minimum distance between them
        
        threshold = height                  # * np.max(velocity_array[:, i])
        peaks, _ = find_peaks(velocity_array[:, i], height= threshold, distance=distance)
        binary_onset_data = np.zeros(len(velocity_array[:, i]))
        binary_onset_data[peaks] = 1        # directional change onsets represented by 1
        
        dir_change_onset_arr.append(binary_onset_data)

    dir_change_onset_arr = np.column_stack(dir_change_onset_arr)
    return dir_change_onset_arr


def detrend_signal(signal, cutoff= 0.5, fs=60):
    
    b, a = butter(2, cutoff / (fs / 2), btype='highpass')
    detrended_signal = filtfilt(b, a, signal) 
    return detrended_signal


def detrend_signal_array(signal, cutoff= 0.5, fs=60):
  
    b, a = butter(2, cutoff / (fs / 2), btype='highpass')
    
    detrended_array = np.array([])
    detrended_list = []
    for i in range(signal.shape[1]):
        detrended_signal = filtfilt(b, a, signal[:,i]) 
        detrended_list.append(detrended_signal)
    detrended_array = np.column_stack(detrended_list)
    
    return detrended_array

def calc_xy_yz_zx(sensor_velocity):
    xy = np.sqrt(sensor_velocity[:,0]**2 + sensor_velocity[:,1]**2).flatten()
    yz = np.sqrt(sensor_velocity[:,1]**2 + sensor_velocity[:,2]**2).flatten()
    xz = np.sqrt(sensor_velocity[:,0]**2 + sensor_velocity[:,2]**2).flatten() 
    arr = np.column_stack([xy,yz,xz])
    return arr

def calc_xyz(sensor_velocity):
    xyz = np.sqrt(sensor_velocity[:,0]**2 + sensor_velocity[:,1]**2+ sensor_velocity[:,2]**2).flatten()
    return xyz.reshape(-1,1)

def calculate_hand_distance_vectors(right_hand_velocity, left_hand_velocity):
    """
    Calculate the velocity vector for each axis between right and left hand velocities.
    
    Parameters:
    - right_hand_velocity: ndarray of shape (n, 3), representing right hand velocities (x, y, z)
    - left_hand_velocity: ndarray of shape (n, 3), representing left hand velocities (x, y, z)
    
    Returns:
    - distance_vectors: ndarray of shape (n, 3), representing distance vectors for each axis (x, y, z)
    """
    # Calculate the absolute difference for each axis (x, y, z)
    x_distance_vector = np.abs(right_hand_velocity[:, 0] - left_hand_velocity[:, 0])
    y_distance_vector = np.abs(right_hand_velocity[:, 1] - left_hand_velocity[:, 1])
    z_distance_vector = np.abs(right_hand_velocity[:, 2] - left_hand_velocity[:, 2])

    # Stack the distances into an (n, 3) array for all axes
    distance_vectors = np.column_stack((x_distance_vector, y_distance_vector, z_distance_vector))
    
    return distance_vectors

def z_score_normalize(data):
    mean_vals = np.mean(data, axis=0)  # Mean values along each column
    std_vals = np.std(data, axis=0)   # Standard deviation along each column
    normalized_data = (data - mean_vals) / std_vals
    return normalized_data

# Min-Max Normalization
def min_max_normalize(data):
    min_vals = np.min(data, axis=0)  # Minimum values along each column
    max_vals = np.max(data, axis=0)  # Maximum values along each column
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def binary_to_peak(binary_array, sampling_rate=60, peak_duration=0.05):
    """
    Represent binary array with 50ms bandwidth peaks.

    Parameters:
        binary_array (numpy.ndarray): Binary input array (1s and 0s).
        sampling_rate (int): Sampling rate in Hz.
        peak_duration (float): Duration of each peak in seconds (default is 0.05s).

    Returns:
        numpy.ndarray: Continuous signal with peaks for each 1 in the binary array.
    """
    n = len(binary_array)
    peak_samples = int(peak_duration * sampling_rate)  # Samples in 50ms
    half_peak = peak_samples // 2

    # Create a Gaussian peak
    t = np.linspace(-half_peak, half_peak, peak_samples)
    peak_shape = np.exp(-0.5 * (t / (half_peak / 2))**2)  # Gaussian

    # Create the continuous signal
    continuous_signal = np.zeros(n + peak_samples)
    for i, value in enumerate(binary_array):
        if value == 1:
            start = i
            end = i + peak_samples
            continuous_signal[start:end] += peak_shape

    # Trim to original length
    return continuous_signal[:n].reshape(-1,1)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        json_data = pickle.load(f)
    return json_data

def save_to_pickle(filepath, data):
    # filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)