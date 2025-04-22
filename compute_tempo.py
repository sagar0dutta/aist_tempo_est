import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter, argrelmax


def moving_average(signal, window_size):
    """
    Compute the moving average of a 1D signal using a specified window size.
    
    Parameters:
        signal (array-like): The input 1D signal.
        window_size (int): The number of samples for the moving average.
    
    Returns:
        np.ndarray: The moving average of the signal.
    """
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')






def compute_tempogram(dir_change_onset, sampling_rate, window_length, hop_size, tempi=np.arange(30, 121, 1)):
    
    tempogram_raw = []
    tempogram_ab = []
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
                
        tempogram_raw.append(tempogram)
        tempogram_ab.append(np.abs(tempogram))
    # print("Tempograms generated")
 
    return tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm


def plot_tempogram(tempo_json, islog= 'no', dpi=200):

    tempogram_ab = tempo_json["tempogram_ab"]
    time_axis_seconds = tempo_json["time_axis_seconds"]
    tempo_axis_bpm = tempo_json["tempo_axis_bpm"]
    # tempogram_ab = np.log(tempogram_ab)
    # tempogram_ab[0][tempogram_ab[0] <= 50] = 0
    # tempogram_ab[1][tempogram_ab[1] <= 50] = 0
    # tempogram_ab[2][tempogram_ab[2] <= 50] = 0
    
    if islog == 'yes':
        tempogram_ab = np.log(tempogram_ab)
    else:
        pass

    fig, axs = plt.subplots(1, 4, figsize=(30,6), dpi=dpi)

    # Tempogram X
    cax1 = axs[0].pcolormesh(time_axis_seconds, tempo_axis_bpm, tempogram_ab[0], shading='auto', cmap='magma')
    axs[0].set_title('X-axis')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Tempo [BPM]')
    plt.colorbar(cax1, ax=axs[0], orientation='horizontal', label='Magnitude')

    # Tempogram Y
    cax2 = axs[1].pcolormesh(time_axis_seconds, tempo_axis_bpm, tempogram_ab[1], shading='auto', cmap='magma')
    axs[1].set_title('Y-axis')
    axs[1].set_xlabel('Time [s]')
    plt.colorbar(cax2, ax=axs[1], orientation='horizontal', label='Magnitude')

    # Tempogram Z
    cax3 = axs[2].pcolormesh(time_axis_seconds, tempo_axis_bpm, tempogram_ab[2], shading='auto', cmap='magma')
    axs[2].set_title('Z-axis')
    axs[2].set_xlabel('Time [s]')
    plt.colorbar(cax3, ax=axs[2], orientation='horizontal', label='Magnitude')

    # Tempogram XYZ
    cax3 = axs[3].pcolormesh(time_axis_seconds, tempo_axis_bpm, (tempogram_ab[0]+tempogram_ab[1]+tempogram_ab[2]), shading='auto', cmap='magma')
    axs[3].set_title('XYZ-axis')
    axs[3].set_xlabel('Time [s]')
    plt.colorbar(cax3, ax=axs[3], orientation='horizontal', label='Magnitude')

    # plt.suptitle(f'{segment_name} tempograms for the 3 axes')
    plt.show()

def plot_tempogram_perAxis(tempo_json, islog= 'no', dpi=100):

    tempogram_ab = tempo_json["tempogram_ab"]
    time_axis_seconds = tempo_json["time_axis_seconds"]
    tempo_axis_bpm = tempo_json["tempo_axis_bpm"]
    
    if islog == 'yes':
        tempogram_ab = np.log(tempogram_ab)
    else:
        pass
    
    Tdim = len(tempogram_ab)
    fig, axs = plt.subplots(1, 1, figsize=(5,5), dpi=dpi)

    for i in range(Tdim):
        # Tempogram X
        cax1 = axs.pcolormesh(time_axis_seconds, tempo_axis_bpm, tempogram_ab[i], shading='auto', cmap='magma')
        axs.set_title(f'{i}-axis')
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Tempo [BPM]')
        plt.colorbar(cax1, ax=axs, orientation='horizontal', label='Magnitude')
    plt.show()

def dance_beat_tempo_estimation_maxmethod(tempogram_ab, tempogram_raw, sampling_rate, novelty_length, window_length, hop_size, tempi):
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
    left_padding = half_window_length
    right_padding = half_window_length
    padded_curve_length = novelty_length + left_padding + right_padding
    estimated_beat_pulse = np.zeros(padded_curve_length)
    num_frames = tempogram_raw[0].shape[1]

    tempo_curve = np.zeros(padded_curve_length)
    tempo_curve_taxis = np.linspace(0, novelty_length/sampling_rate, novelty_length)
    prev_freq = None
    mag_list = []
    phase_list = []
    bpm_list = []
    for frame_idx in range(num_frames):
        
        bpm_arr = np.array([])
        freq_arr = np.array([])
        mag_arr = np.array([])
        phase_arr = np.array([])
        
        for i in range(len(tempogram_ab)):  # number of axis = 3 
        
            # select peak frequency for a time window
            peak_tempo_idx = np.argmax(tempogram_ab[i][:, frame_idx])
            peak_tempo_bpm = tempi[peak_tempo_idx]
            frequency = (peak_tempo_bpm / 60) / sampling_rate
            frequency = np.round(frequency, 3)

            # get the complex value for that peak frequency and time window
            complex_value = tempogram_raw[i][peak_tempo_idx, frame_idx]
            phase = - np.angle(complex_value) / (2 * np.pi)
            magnitude = np.abs(complex_value)
            
            start_index = frame_idx * hop_size
            end_index = start_index + window_length
            time_kernel = np.arange(start_index, end_index)
            
            freq_arr = np.concatenate(( freq_arr, np.array([frequency]) ))
            bpm_arr = np.concatenate(( bpm_arr, np.array([peak_tempo_bpm]) ))
            mag_arr = np.concatenate(( mag_arr, np.array([magnitude]) ))
            phase_arr = np.concatenate(( phase_arr, np.array([phase]) ))
        
        # f_idx = np.argmax(freq_arr)     # selects the peak frequency for the time window
        # selected_freq = freq_arr[f_idx]
        # selected_bpm = bpm_arr[f_idx]
        # selected_mag = mag_arr[f_idx]
        # selected_phase = phase_arr[f_idx]
        
        m_idx = np.argmax(mag_arr)     # selects the peak magnitude for the time window
        selected_freq = freq_arr[m_idx]
        selected_bpm = bpm_arr[m_idx]
        selected_mag = mag_arr[m_idx]
        selected_phase = phase_arr[m_idx]
        
        
        mag_list.append(selected_mag)
        bpm_list.append(selected_bpm)   # bpm per window
        
        sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * selected_freq - selected_phase))
        estimated_beat_pulse[time_kernel] += sinusoidal_kernel

    tempo_curve = tempo_curve[left_padding:padded_curve_length-right_padding]
    
    estimated_beat_pulse = estimated_beat_pulse[left_padding:padded_curve_length-right_padding]
    estimated_beat_pulse[estimated_beat_pulse < 0] = 0

    json_data = {"estimated_beat_pulse": estimated_beat_pulse,
                 "tempo_curve": tempo_curve,
                 "tempo_curve_time_axis": tempo_curve_taxis,
                 "mag_arr": np.array(mag_list),
                 "bpm_arr": np.array(bpm_list),}

    return json_data

def dance_beat_tempo_estimation_topN(tempogram_ab, tempogram_raw, sampling_rate, novelty_length, window_length, hop_size, tempi):
    """Compute windowed sinusoid with optimal phase


    Args:
        tempogram (np.ndarray): Fourier-based (complex-valued) tempogram
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
    left_padding = half_window_length
    right_padding = half_window_length
    padded_curve_length = novelty_length + left_padding + right_padding
    estimated_beat_pulse = np.zeros(padded_curve_length)
    num_frames = tempogram_raw[0].shape[1]

    tempo_curve = np.zeros(padded_curve_length)
    tempo_curve_taxis = np.linspace(0, novelty_length/sampling_rate, novelty_length)


    freq_all_frames = []
    bpm_all_frames = []
    for frame_idx in range(num_frames):
        
        
        freq_arr_axes = []
        bpm_arr_axes = []
        for i in range(len(tempogram_ab)):  # number of axis = 3 
        
            n = 10  # Number of top peaks you want
            peak_tempo_indices = np.argsort(tempogram_ab[i][:, frame_idx])[-n:]  # Indices of top n values, sorted in ascending order
            peak_tempo_indices = peak_tempo_indices[::-1]  # Reverse to get them in descending order
            
            top_n_freq_arr = np.array([])
            top_n_bpm_arr = np.array([])
            for idx in peak_tempo_indices:  # Iterate over the top n indices
                peak_tempo_bpm = tempi[idx]
                frequency = (peak_tempo_bpm / 60) / sampling_rate
                peak_frequency = np.round(frequency, 3)
            
                # Get the complex value for that peak frequency and time window
                complex_value = tempogram_raw[i][idx, frame_idx]
                magnitude = np.abs(complex_value)
                phase = - np.angle(complex_value) / (2 * np.pi)
            
                start_index = frame_idx * hop_size
                end_index = start_index + window_length
                time_kernel = np.arange(start_index, end_index)
                
                top_n_bpm_arr = np.concatenate(( top_n_bpm_arr, np.array([peak_tempo_bpm]) ))
                top_n_freq_arr = np.concatenate(( top_n_freq_arr, np.array([peak_frequency]) ))
                
            freq_arr_axes.append(top_n_freq_arr)
            bpm_arr_axes.append(top_n_bpm_arr) # list of array of top n bpm array for the three axes
          
        freq_all_frames.append(np.column_stack(freq_arr_axes))
        bpm_all_frames.append(np.column_stack(bpm_arr_axes))
        
        frame_bpm = np.column_stack(bpm_arr_axes)
        top_n_max_bpm = np.argmax(frame_bpm, axis=0)        # 1d array

        # Calculate the weighted BPM and frequency for the top n values
        top_n_bpm = frame_bpm.flatten()  # Flatten the array for easy manipulation
        top_n_freq = np.array([bpm / 60 / sampling_rate for bpm in top_n_bpm])
        top_n_magnitudes = np.abs(tempogram_raw[i][peak_tempo_indices, frame_idx])  # Corresponding magnitudes

        if np.sum(top_n_magnitudes) > 0:
            # Weighted BPM and frequency
            weighted_bpm = np.sum(top_n_bpm * top_n_magnitudes) / np.sum(top_n_magnitudes)
            weighted_freq = np.sum(top_n_freq * top_n_magnitudes) / np.sum(top_n_magnitudes)
        else:
            weighted_bpm = 0
            weighted_freq = 0

        # Use the weighted BPM and frequency for tempo curve and sinusoidal kernel
        selected_bpm = weighted_bpm
        selected_freq = weighted_freq

        tempo_curve[time_kernel] = selected_bpm
        sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * selected_freq - phase))
        estimated_beat_pulse[time_kernel] += sinusoidal_kernel

    json_data = {"estimated_beat_pulse": estimated_beat_pulse,
                 "tempo_curve": tempo_curve,
                 "tempo_curve_time_axis": tempo_curve_taxis,
                 "bpm_arr": bpm_all_frames,}

    return json_data

def dance_beat_tempo_estimation_weightedkernelmethod(tempogram_ab, tempogram_raw, sampling_rate, novelty_length, window_length, hop_size, tempi):
    """Overlapping kernels from all axis per frame
    Args:
        tempogram (np.ndarray): Fourier-based (complex-valued) tempogram
        sampling_rate (scalar): Sampling rate
        novelty_length (int): Length of novelty curve
        window_length (int): Window length
        hop_size (int): Hop size
        tempi (np.ndarray): Set of tempi (given in BPM)

    Returns:
        predominant_local_pulse (np.ndarray): PLP function
    """

    hann_window = np.hanning(window_length)
    half_window_length = window_length // 2
    left_padding = half_window_length
    right_padding = half_window_length
    padded_curve_length = novelty_length + left_padding + right_padding
    estimated_beat_pulse = np.zeros(padded_curve_length)
    num_frames = tempogram_raw[0].shape[1]

    tempo_curve = np.zeros(padded_curve_length)
    tempo_curve_taxis = np.linspace(0, len(tempo_curve)/sampling_rate, len(tempo_curve))
    bpm_list = []
    for frame_idx in range(num_frames):
        
        bpm_arr = np.array([])
        magnitude_arr = np.array([])
        weighted_kernel_sum = np.zeros(window_length)
        total_weight = 0
        
        for i in range(len(tempogram_ab)):  # number of axis = 3 
        
            # select peak frequency for a time window
            peak_tempo_idx = np.argmax(tempogram_ab[i][:, frame_idx])
            peak_tempo_bpm = tempi[peak_tempo_idx]
            frequency = (peak_tempo_bpm / 60) / sampling_rate
            
            # get the complex value for that peak frequency and time window
            complex_value = tempogram_raw[i][peak_tempo_idx, frame_idx]
            magnitude = np.abs(complex_value)
            phase = - np.angle(complex_value) / (2 * np.pi)
            
            start_index = frame_idx * hop_size
            end_index = start_index + window_length
            time_kernel = np.arange(start_index, end_index)
            
            magnitude_arr = np.concatenate(( magnitude_arr, np.array([magnitude]) ))
            bpm_arr = np.concatenate(( bpm_arr, np.array([peak_tempo_bpm]) ))     #  

            sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * frequency - phase))
            weighted_kernel_sum += magnitude * sinusoidal_kernel   
            total_weight += magnitude

        if total_weight > 0:
            weighted_kernel_sum /= total_weight     # Normalize

        estimated_beat_pulse[time_kernel] += weighted_kernel_sum

        if len(bpm_arr) > 0:
            # selected_bpm = np.max(bpm_arr)          # mean median not good, max is good
            if np.sum(magnitude_arr) == 0:
                selected_bpm = 0
            else:
                selected_bpm = np.sum(bpm_arr * magnitude_arr) / np.sum(magnitude_arr)
                tempo_curve[time_kernel] = selected_bpm
        else:
            selected_bpm = 0
        bpm_list.append(selected_bpm)
        
    tempo_curve = tempo_curve[left_padding:padded_curve_length-right_padding]
    
    estimated_beat_pulse = estimated_beat_pulse[left_padding:padded_curve_length-right_padding]
    estimated_beat_pulse[estimated_beat_pulse < 0] = 0

    json_data = {"estimated_beat_pulse": estimated_beat_pulse,
                 "tempo_curve": tempo_curve,
                 "tempo_curve_time_axis": tempo_curve_taxis,
                #  "global_tempo_bpm": global_bpm,
                 "bpm_arr": np.array(bpm_list)}
    
        # weighted average BPM for the tempo curve
        # if np.sum(magnitude_arr) > 0:
        #     bpm_weighted_sum = np.sum(bpm_arr * magnitude_arr)
        #     avg_bpm = bpm_weighted_sum / np.sum(magnitude_arr)
        # else:
        #     avg_bpm = 0
        # tempo_curve[time_kernel] = avg_bpm
        
        # Using median bpm
    return json_data


def dance_beat_tempo_estimation_combinedtempogram_method(tempogram_ab, tempogram_raw, sampling_rate, novelty_length, window_length, hop_size, tempi):
    """
    Args:
        tempogram (np.ndarray): Fourier-based (complex-valued) tempogram
        sampling_rate (scalar): Sampling rate
        novelty_length (int): Length of novelty curve
        window_length (int): Window length
        hop_size (int): Hop size
        tempi (np.ndarray): Set of tempi (given in BPM)

    Returns:
        predominant_local_pulse (np.ndarray): PLP function
    """

    hann_window = np.hanning(window_length)
    half_window_length = window_length // 2
    left_padding = half_window_length
    right_padding = half_window_length
    padded_curve_length = novelty_length + left_padding + right_padding
    estimated_beat_pulse = np.zeros(padded_curve_length)
    num_frames = tempogram_raw[0].shape[1]

    tempo_curve = np.zeros(padded_curve_length)
    tempo_curve_taxis = np.linspace(0, len(tempo_curve)/sampling_rate, len(tempo_curve))
    bpm_list = []
    for frame_idx in range(num_frames):
        
        bpm_arr = np.array([])
        freq_arr = np.array([])
        phase_arr = np.array([])
        
        for i in range(len(tempogram_ab)):  # number of axis = 3 
        
            # select peak frequency for a time window
            peak_tempo_idx = np.argmax(tempogram_ab[i][:, frame_idx])
            peak_tempo_bpm = tempi[peak_tempo_idx]
            frequency = (peak_tempo_bpm / 60) / sampling_rate

            # get the complex value for that peak frequency and time window
            complex_value = tempogram_raw[i][peak_tempo_idx, frame_idx]
            phase = - np.angle(complex_value) / (2 * np.pi)
            
            start_index = frame_idx * hop_size
            end_index = start_index + window_length
            time_kernel = np.arange(start_index, end_index)
            
            freq_arr = np.concatenate(( freq_arr, np.array([frequency]) ))
            bpm_arr = np.concatenate(( bpm_arr, np.array([peak_tempo_bpm]) ))
            phase_arr = np.concatenate(( phase_arr, np.array([phase]) ))

        f_idx = np.argmax(freq_arr)
        selected_freq = freq_arr[f_idx]
        selected_bpm = bpm_arr[f_idx]
        selected_phase = phase_arr[f_idx]
        bpm_list.append(selected_bpm)
        
        sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * selected_freq - selected_phase))
        estimated_beat_pulse[time_kernel] += sinusoidal_kernel
        
        tempo_curve[time_kernel] = selected_bpm
        
    # global_bpm = np.average(tempo_curve)
        
    estimated_beat_pulse = estimated_beat_pulse[left_padding:padded_curve_length-right_padding]
    estimated_beat_pulse[estimated_beat_pulse < 0] = 0

    json_data = {"estimated_beat_pulse": estimated_beat_pulse,
                 "tempo_curve": tempo_curve,
                 "tempo_curve_time_axis": tempo_curve_taxis,
                #  "global_tempo_bpm": global_bpm,
                 "bpm_array": bpm_list}

    return json_data




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


# def filter_onsets_by_distance(xyz_ab_minima, xyz_ab, distance_threshold=0.1, time_threshold=0, fps=60):
    
#     # xyz_ab_minima: minima from the velocity data, xyz_ab: velocity data
#     filtered_onsets = []
    
#     # Iterate through the onsets
#     for i in range(len(xyz_ab_minima) - 1):
#         onset_current = xyz_ab_minima[i]
#         onset_next = xyz_ab_minima[i + 1]
        
#         # Calculate the distance between the two onsets (in terms of velocity)
#         distance = np.sum(np.abs(xyz_ab[onset_current:onset_next])) / fps
        
#         # Compute time difference in frames
#         time_diff = (onset_next-onset_current)/fps
        
#         # Apply the distance threshold
#         if distance > distance_threshold and time_diff >= time_threshold:
#             # Keep the next onset
#             filtered_onsets.append(onset_next)
    
#     return np.array(filtered_onsets)

# def velocity_based_novelty(velocity_array, order=15):
    
#     dir_change_onset_arr = np.array([])
#     onset_data_list = []
#     for i in range(velocity_array.shape[1]):
        
#         maxima_indices = argrelmax(velocity_array[:,i], order=order)[0]
#         binary_onset_data = np.zeros(len(velocity_array[:,i]))
#         binary_onset_data[maxima_indices] = 1                      # directional change onsets represented by value 1

#         onset_data_list.append(binary_onset_data)
#     dir_change_onset_arr = np.column_stack(onset_data_list)
    
#     return dir_change_onset_arr

##### april 2025
def velocity_based_novelty(velocity_array, height = 0.2, distance=15):
    dir_change_onset_arr = []
    
    
    for i in range(velocity_array.shape[1]):
        # Find peaks (local maxima) with a minimum distance between them
        
        threshold = height # * np.max(velocity_array[:, i])
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