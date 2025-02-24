import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter, argrelmax

def main_two_sensor(sensorA_velocity, sensorB_velocity, 
                    mocap_fps, window_size, hop_size, tempi_range, 
                    T_filter= 0.25, smooth_wlen= 10, pk_order = 15, vel_mode = "on",mode= "zero_uni"):
    # to used for any combincation of two sensors or two body markers

    novelty_length = len(sensorA_velocity)
    
    if mode== "zero_uni":
        sensorA_abs_vel = smooth_velocity(sensorA_velocity, abs="no", window_length = smooth_wlen, polyorder = 0)
        sensorB_abs_vel = smooth_velocity(sensorB_velocity, abs="no", window_length = smooth_wlen, polyorder = 0) # size (n, 3)
        if vel_mode == "on":
            sensorA_abs_vel = np.diff(sensorA_abs_vel, axis=0)    # velocity
            sensorB_abs_vel = np.diff(sensorB_abs_vel, axis=0)    # velocity
        
        sensorA_abs_vel[sensorA_abs_vel < 0] = 0
        sensorB_abs_vel[sensorB_abs_vel < 0] = 0
        
        sensorA_dir_change = velocity_based_novelty(sensorA_abs_vel, order=pk_order)    # size (n, 3)   binary
        sensorB_dir_change = velocity_based_novelty(sensorB_abs_vel, order=pk_order)    # size (n, 3)   binary
        sensorA_sensorB_dir_change = sensorA_dir_change + sensorB_dir_change      # Merge axis wise
        
    elif mode == "zero_bi":
        sensorA_abs_vel = smooth_velocity(sensorA_velocity, abs="yes", window_length = smooth_wlen, polyorder = 0)
        sensorB_abs_vel = smooth_velocity(sensorB_velocity, abs="yes", window_length = smooth_wlen, polyorder = 0) # size (n, 3)
        
        if vel_mode == "on":
            sensorA_abs_vel = np.diff(sensorA_abs_vel, axis=0)    # velocity
            sensorB_abs_vel = np.diff(sensorB_abs_vel, axis=0)    # velocity
        
        sensorA_dir_change = velocity_based_novelty(sensorA_abs_vel, order=pk_order)    # size (n, 3)   binary
        sensorB_dir_change = velocity_based_novelty(sensorB_abs_vel, order=pk_order)    # size (n, 3)   binary
        sensorA_sensorB_dir_change = sensorA_dir_change + sensorB_dir_change      # Merge axis wise


    sensorA_sensorB_dir_change = np.where(sensorA_sensorB_dir_change > 0, 1,0)    # make onset values to 1
    both_sensor_onsets = filter_dir_onsets_by_threshold(sensorA_sensorB_dir_change, threshold_s= T_filter)   # Filter onsets axis wise 
    
    both_sensor_onsets = binary_to_peak(both_sensor_onsets, peak_duration=0.05)
    
    # print("Computing tempograms...")
    tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(both_sensor_onsets, mocap_fps, 
                                                                window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    
    # print("Computing max method...")
    tempo_data_maxmethod = dance_beat_tempo_estimation_maxmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                   novelty_length, window_size, hop_size, tempi_range)
    # print("Computing weighted method...")
    tempo_data_weightedkernel = dance_beat_tempo_estimation_weightedkernelmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                 novelty_length, window_size, hop_size, tempi_range)
    # print("Computing combined method...")
    # tempo_data_combinedtempogram = dance_beat_tempo_estimation_combinedtempogram_method([tempogram_ab[0]+tempogram_ab[1]+tempogram_ab[2]], 
                                                                                    # [tempogram_raw[0]+tempogram_raw[1]+tempogram_raw[2]],                                                                                     # mocap_fps, novelty_length, window_size, hop_size, tempi_range)
    tempo_data_combinedtempogram = None
    
    json_tempodata = {
        "sensorA_abs_vel": sensorA_abs_vel, 
        "sensorB_abs_vel": sensorB_abs_vel,
        "sensorA_dir_change": sensorA_dir_change, 
        "sensorB_dir_change": sensorB_dir_change,
        # "sensorA_dir_change_f": sensorA_dir_change_f, 
        # "sensorB_dir_change_f": sensorB_dir_change_f,
    
        "sensorAB_onsets": both_sensor_onsets,
        
        "tempogram_ab": tempogram_ab,
        "tempogram_raw": tempogram_raw,
        "time_axis_seconds": time_axis_seconds,
        "tempo_axis_bpm": tempo_axis_bpm,
        
        "tempo_data_maxmethod": tempo_data_maxmethod,
        "tempo_data_weightedkernel": tempo_data_weightedkernel,
        "tempo_data_combinedtempogram": tempo_data_combinedtempogram
    }
    
    return json_tempodata


def main_one_sensor_peraxis(sensor_position, mocap_fps, window_size, 
                            hop_size, tempi_range, T_filter=0.25, 
                            smooth_wlen= 100, pk_order = 30,vel_mode="on",mode = "zero"):
    # to used for any combincation of two sensors or two body markers
    sensor_dir_change = None
    sensor_dir_change_f = None
    sensor_onsets = None  
    novelty_length = len(sensor_position)
    
    if mode == 'zero_uni':          # Extract uni-directional change onsets
        
        sensor_abs_pos = smooth_velocity(sensor_position, abs="no", window_length = smooth_wlen, polyorder = 0) # size (n, 3)
        if vel_mode == "on":
            sensor_abs_pos = np.diff(sensor_abs_pos, axis=0)    # velocity
        
        sensor_abs_pos[sensor_abs_pos < 0] = 0
        
        sensor_dir_change = velocity_based_novelty(sensor_abs_pos, order= pk_order)    # size (n, 3)
        sensor_onsets = filter_dir_onsets_by_threshold(sensor_dir_change, threshold_s= T_filter)
        
        tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onsets, mocap_fps, 
                                                                    window_length=window_size, hop_size=hop_size, tempi=tempi_range)

    elif mode == 'zero_bi':         # Extract bi-directional change onsets
        
        sensor_abs_pos = smooth_velocity(sensor_position, abs="yes", window_length = smooth_wlen, polyorder = 0) # size (n, 3)
        if vel_mode == "on":
            sensor_abs_pos = np.diff(sensor_abs_pos, axis=0)   # velocity
        
        sensor_dir_change = velocity_based_novelty(sensor_abs_pos, order=pk_order)    # size (n, 3)
        sensor_onsets = filter_dir_onsets_by_threshold(sensor_dir_change, threshold_s= T_filter)
        
        tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onsets, mocap_fps, 
                                                                    window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    

    tempo_data_maxmethod = dance_beat_tempo_estimation_maxmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                   novelty_length, window_size, hop_size, tempi_range)
    
    tempo_data_weightedkernel = dance_beat_tempo_estimation_weightedkernelmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                 novelty_length, window_size, hop_size, tempi_range)
    
    tempo_data_topN = dance_beat_tempo_estimation_topN(tempogram_ab, tempogram_raw, mocap_fps, 
                                                   novelty_length, window_size, hop_size, tempi_range)

    json_tempodata = {
        "sensor_abs_vel": sensor_abs_pos,
        "sensor_dir_change_onsets": sensor_dir_change,
        "sensor_dir_change_onsets_f": sensor_dir_change_f,
        "sensor_onsets": sensor_onsets,

        "tempogram_ab": tempogram_ab,
        "tempogram_raw": tempogram_raw,
        "time_axis_seconds": time_axis_seconds,
        "tempo_axis_bpm": tempo_axis_bpm,
        
        "tempo_data_maxmethod": tempo_data_maxmethod,
        "tempo_data_weightedkernel": tempo_data_weightedkernel,
        "tempo_data_topN": tempo_data_topN,
    }
    
    return json_tempodata




def get_peak_onsets(sensorA_velocity_ax, height= 0.02, distance = None, percentile = 40):
    
    peaks_temp, _ = find_peaks(sensorA_velocity_ax.flatten(), height= height, distance= distance)
    peak_values = sensorA_velocity_ax[peaks_temp]
    h_cutoff = np.percentile(peak_values, percentile)

    peaks, _ = find_peaks(sensorA_velocity_ax.flatten(), height= h_cutoff, prominence= h_cutoff)
    peak_onset = np.zeros(len(sensorA_velocity_ax))
    peak_onset[peaks] = 1
    
    return peak_onset.reshape(-1,1)

def main_two_sensor_feet(sensorA_velocity_znorm, sensor_position,both_sensor_onsets, mocap_fps, window_size, 
                         hop_size, tempi_range, mode="peak_uni", ax=2):
    
    novelty_length = len(both_sensor_onsets)
    
    # uni directional peak onsets
    if mode == 'zero_uni':          # Extract uni-directional change onsets
        sensorA_velocity_ax = sensorA_velocity_znorm[:, ax].reshape(-1,1)
        sensorA_position_ax = sensor_position[:, ax].reshape(-1,1)
        
        sensor_abs_vel = smooth_velocity(sensorA_velocity_ax, abs="yes", window_length = 60, polyorder = 0) # size (n, 3)
        sensor_dir_change = velocity_based_novelty(sensor_abs_vel, distance_threshold=0.015, time_threshold=0, vel_threshold= 0.05)    # size (n, 3)
        sensor_dir_change_f = filter_velocity_by_position(sensorA_position_ax, sensor_dir_change)   # uni direction filter
        sensor_onset = filter_dir_onsets_by_threshold(sensor_dir_change_f, threshold_s= 0.25)

        # tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onsets, mocap_fps, 
        #                                                                 window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    
    elif mode == 'zero_bi':         # Extract bi-directional change onsets 
        sensorA_velocity_ax = sensorA_velocity_znorm[:, ax].reshape(-1,1)   
        sensor_abs_vel = smooth_velocity(sensorA_velocity_ax, abs="yes", window_length = 60, polyorder = 0) # size (n, 3)
        sensor_dir_change = velocity_based_novelty(sensor_abs_vel, distance_threshold=0.015, time_threshold=0, vel_threshold= 0.05)    # size (n, 3)
        sensor_onset = filter_dir_onsets_by_threshold(sensor_dir_change, threshold_s= 0.25)
        
        # tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onsets, mocap_fps, 
        #                                                             window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    
    elif mode== "peak_uni":
        sensor_abs_vel = smooth_velocity(sensorA_velocity_znorm, abs="no", window_length = 60, polyorder = 0 ) # size (n, 3)
        sensor_abs_vel[sensor_abs_vel < 0] = 0
        sensorA_velocity_mmnorm = min_max_normalize(sensor_abs_vel)     # normalize between 0-1
        sensorA_velocity_ax = sensorA_velocity_mmnorm[:, ax]
        sensor_onset = get_peak_onsets(sensorA_velocity_ax, percentile= 40, distance= 50)
        
    # bi directional peak onsets
    elif mode== "peak_bi":
        sensor_abs_vel = smooth_velocity(sensorA_velocity_znorm, abs="yes", window_length = 60, polyorder = 0 ) # size (n, 3)
        sensorA_velocity_mmnorm = min_max_normalize(sensor_abs_vel)
        sensorA_velocity_ax = sensorA_velocity_mmnorm[:, ax]
        sensor_onset = get_peak_onsets(sensorA_velocity_ax, percentile= 40)

    # using both foot onsets
    elif mode=="zero_bothfeet":
        sensor_onset = both_sensor_onsets
        
        
    print("Computing tempograms...")
    tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onset, mocap_fps, 
                                                        window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    print("Computing max method...")
    tempo_data_maxmethod = dance_beat_tempo_estimation_maxmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                   novelty_length, window_size, hop_size, tempi_range)
    print("Computing weighted method...")
    tempo_data_weightedkernel = dance_beat_tempo_estimation_weightedkernelmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                 novelty_length, window_size, hop_size, tempi_range)
    # print("Computing combined method...")
    # tempo_data_combinedtempogram = dance_beat_tempo_estimation_combinedtempogram_method([tempogram_ab[0]+tempogram_ab[1]+tempogram_ab[2]], 
    #                                                                                 [tempogram_raw[0]+tempogram_raw[1]+tempogram_raw[2]], 
    #                                                                                 mocap_fps, novelty_length, window_size, hop_size, tempi_range)
    
    json_tempodata = {
        "feet_sensor_onsets": both_sensor_onsets,
        
        "tempogram_ab": tempogram_ab,
        "tempogram_raw": tempogram_raw,
        "time_axis_seconds": time_axis_seconds,
        "tempo_axis_bpm": tempo_axis_bpm,
        
        "tempo_data_maxmethod": tempo_data_maxmethod,
        "tempo_data_weightedkernel": tempo_data_weightedkernel,

    }
    
    return json_tempodata


def main_one_sensor(sensor_velocity, sensor_position, mocap_fps, window_size, 
                    hop_size, tempi_range, distance_threshold=0.1, vel_thres = 0.05, mode='zero_bi'):
    # to used for any combincation of two sensors or two body markers
    sensor_dir_change = None
    sensor_dir_change_f = None
    sensor_onsets = None    
    novelty_length = len(sensor_velocity)
    # sensor_abs_vel[sensor_abs_vel<0] = 0
    
    if mode == 'zero_uni':          # Extract uni-directional change onsets    
        sensor_abs_vel = smooth_velocity(sensor_velocity, abs="yes", window_length = 60, polyorder = 0) # size (n, 3)
        sensor_dir_change = velocity_based_novelty(sensor_abs_vel, distance_threshold=distance_threshold, time_threshold=0, vel_threshold= vel_thres)    # size (n, 3)
        sensor_dir_change_f = filter_velocity_by_position(sensor_position, sensor_dir_change)   # uni direction filter
        sensor_onsets = filter_dir_onsets_by_threshold(sensor_dir_change_f, threshold_s= 0.25)

        tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onsets, mocap_fps, 
                                                                        window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    
    elif mode == 'zero_bi':         # Extract bi-directional change onsets    
        sensor_abs_vel = smooth_velocity(sensor_velocity, abs="yes", window_length = 60, polyorder = 0) # size (n, 3)
        sensor_dir_change = velocity_based_novelty(sensor_abs_vel, distance_threshold=distance_threshold, time_threshold=0, vel_threshold= vel_thres)    # size (n, 3)
        sensor_onsets = filter_dir_onsets_by_threshold(sensor_dir_change, threshold_s= 0.25)
        
        tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onsets, mocap_fps, 
                                                                    window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    
    elif mode == 'p1':              # using continuous velocity positive peaks
        sensor_abs_vel = smooth_velocity(sensor_velocity, abs="no", window_length = 60, polyorder = 0) # size (n, 3)
        sensor_abs_vel[sensor_abs_vel<0] = 0
        tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_abs_vel, mocap_fps, 
                                                                     window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    
    elif mode == 'p2':              # using continuous velocity absolute positive/negative peaks
        sensor_abs_vel = smooth_velocity(sensor_velocity, abs="yes", window_length = 60, polyorder = 0) # size (n, 3)
        tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_abs_vel, mocap_fps, 
                                                                     window_length=window_size, hop_size=hop_size, tempi=tempi_range)
        
    tempo_data_maxmethod = dance_beat_tempo_estimation_maxmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                   novelty_length, window_size, hop_size, tempi_range)
    
    tempo_data_weightedkernel = dance_beat_tempo_estimation_weightedkernelmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                 novelty_length, window_size, hop_size, tempi_range)
    
    tempo_data_combinedtempogram = dance_beat_tempo_estimation_combinedtempogram_method([tempogram_ab[0]+tempogram_ab[1]+tempogram_ab[2]], 
                                                                                    [tempogram_raw[0]+tempogram_raw[1]+tempogram_raw[2]], 
                                                                                    mocap_fps, novelty_length, window_size, hop_size, tempi_range)
    
    json_tempodata = {
        "sensor_abs_vel": sensor_abs_vel,
        "sensor_dir_change": sensor_dir_change,
        "tempogram_ab": tempogram_ab,
        "tempogram_raw": tempogram_raw,
        "time_axis_seconds": time_axis_seconds,
        "tempo_axis_bpm": tempo_axis_bpm,
        
        "tempo_data_maxmethod": tempo_data_maxmethod,
        "tempo_data_weightedkernel": tempo_data_weightedkernel,
        "tempo_data_combinedtempogram": tempo_data_combinedtempogram
    }
    
    return json_tempodata


def get_peak_onsets_1S_xyz(sensorA_velocity_ax, height= 0.02, distance = None, percentile = 20, prominence = 0.3):
    
    peaks_temp, _ = find_peaks(sensorA_velocity_ax.flatten(), height= height, distance= distance)
    peak_values = sensorA_velocity_ax[peaks_temp]
    h_cutoff = np.percentile(peak_values, percentile)

    peaks, _ = find_peaks(sensorA_velocity_ax.flatten(), height= h_cutoff, prominence= prominence)
    peak_onset = np.zeros(len(sensorA_velocity_ax))
    peak_onset[peaks] = 1
    
    return peak_onset.reshape(-1,1)


def main_1S_xyz(sensor_velocity, mocap_fps, window_size, 
                hop_size, tempi_range):
    # to used for any combincation of two sensors or two body markers

    novelty_length = len(sensor_velocity)

    # velocity smoothing
    sensor_abs_vel = smooth_velocity(sensor_velocity, abs="no", window_length = 100, polyorder = 0) # size (n, 3)
    sensor_onsets = get_peak_onsets_1S_xyz(sensor_abs_vel, height= 0.02, distance = 50, percentile = 20, prominence = 0.3)
    
    tempogram_ab, tempogram_raw, time_axis_seconds, tempo_axis_bpm = compute_tempogram(sensor_onsets, mocap_fps, 
                                                                window_length=window_size, hop_size=hop_size, tempi=tempi_range)
    
    tempo_data_maxmethod = dance_beat_tempo_estimation_maxmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                                novelty_length, window_size, hop_size, tempi_range)
    
    tempo_data_weightedkernel = dance_beat_tempo_estimation_weightedkernelmethod(tempogram_ab, tempogram_raw, mocap_fps, 
                                                                novelty_length, window_size, hop_size, tempi_range)
    

    json_tempodata = {
        "sensor_abs_vel": sensor_abs_vel,

        "tempogram_ab": tempogram_ab,
        "tempogram_raw": tempogram_raw,
        "time_axis_seconds": time_axis_seconds,
        "tempo_axis_bpm": tempo_axis_bpm,
        
        "tempo_data_maxmethod": tempo_data_maxmethod,
        "tempo_data_weightedkernel": tempo_data_weightedkernel,
    }
    
    return json_tempodata


def get_zero_onsets(sensor_velocity, sensor_position, 
                    distance_threshold=0.1, tcut =0.25, vel_thres = 0.05,):

    # Extract uni-directional change onsets    
    sensor_abs_vel = smooth_velocity(sensor_velocity, abs="yes", window_length = 60, polyorder = 0) # size (n, 3)
    sensor_dir_change = velocity_based_novelty(sensor_abs_vel, distance_threshold=distance_threshold, time_threshold=0, vel_threshold= vel_thres)    # size (n, 3)
    sensor_dir_change_f = filter_velocity_by_position(sensor_position, sensor_dir_change)   # uni direction filter
    zero_uni_onsets = filter_dir_onsets_by_threshold(sensor_dir_change_f, threshold_s= tcut)


    # Extract bi-directional change onsets    
    sensor_abs_vel1 = smooth_velocity(sensor_velocity, abs="yes", window_length = 60, polyorder = 0) # size (n, 3)
    sensor_dir_change1 = velocity_based_novelty(sensor_abs_vel1, distance_threshold=distance_threshold, time_threshold=0, vel_threshold= vel_thres)    # size (n, 3)
    zero_bi_onsets = filter_dir_onsets_by_threshold(sensor_dir_change1, threshold_s= tcut)
        
    return zero_uni_onsets, zero_bi_onsets
    
    
def filter_velocity_by_position(sensorA_position, sensor_dir_change_onsets):
    """
    Filters velocity based onsets using position data

    Parameters:
    - sensorA_position: ndarray, position data of shape (n, 3).
    - sensor_dir_change_onsets: ndarray, direction change onsets of shape (n, 3).
    - start_f: int, start frame index.
    - end_f: int, end frame index.

    Returns:
    - new_indices: ndarray of shape (n, 3), with filtered indices for each axis.
    """
    new_indices_list = []

    for idx in range(sensorA_position.shape[1]):  # Loop through each axis (0, 1, 2)
        # Create a binary mask indicating positions greater than 0
        pos_b = np.where(sensorA_position[:, idx] > 0, 1, 0)

        # Get indices where direction change onset is greater than 0
        indices = np.where(sensor_dir_change_onsets[:, idx] > 0)[0]

        # Filter indices where the position is also > 0
        filtered_indices = indices[pos_b[indices] == 1]

        padded_indices = np.zeros(len(sensorA_position))
        padded_indices[filtered_indices] = 1

        new_indices_list.append(padded_indices)

    # Stack along columns to create an (n, 3) array
    sensor_dir_change_f = np.column_stack(new_indices_list)

    return sensor_dir_change_f



def compute_tempogram(dir_change_onset, sampling_rate, window_length, hop_size, tempi=np.arange(30, 121, 1)):
    """Compute Fourier-based tempogram [FMP, Section 6.2.2]

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        signal (np.ndarray): Input signal
        sampling_rate (scalar): Sampling rate
        window_length (int): Window length
        hop_size (int): Hop size
        tempi (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601, 1))

    Returns:
        tempogram (np.ndarray): Tempogram
        time_axis_seconds (np.ndarray): Time axis (seconds)
        tempo_axis_bpm (np.ndarray): Tempo axis (BPM)
    """
    
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
        
        f_idx = np.argmax(freq_arr)
        selected_freq = freq_arr[f_idx]
        selected_bpm = bpm_arr[f_idx]
        selected_mag = mag_arr[f_idx]
        selected_phase = phase_arr[f_idx]
        
        mag_list.append(selected_mag)
        bpm_list.append(selected_bpm)   # bpm per window
        
        ################################################
        # margin_bpm = 10        # Small margin for double and half checks (e.g., ±5%)
        # if prev_freq is not None:
        #     # Define bounds for double and half frequencies with margin
        #     lower_double = (2 * prev_freq) - margin_bpm  # Lower bound for double
        #     upper_double = (2 * prev_freq) + margin_bpm  # Upper bound for double
        #     lower_half = (prev_freq / 2) - margin_bpm    # Lower bound for half
        #     upper_half = (prev_freq / 2) + margin_bpm    # Upper bound for half

        #     # Check if selected_freq is close to double or half of prev_freq within the margin
        #     if (lower_double <= selected_freq <= upper_double):
        #         selected_freq = prev_freq  # Retain previous frequency
        #         tempo_curve[time_kernel] = selected_bpm / 2  # Halve the BPM if conditions are met
            
        #     elif (lower_half <= selected_freq <= upper_half):
        #         selected_freq = prev_freq  # Retain previous frequency
        #         tempo_curve[time_kernel] = selected_bpm * 2  # double the BPM if conditions are met         
        #     else:
        #         tempo_curve[time_kernel] = selected_bpm  # Use the full selected BPM
        # prev_freq = selected_freq       # Update previous frequency to the current value
        ######################################################
        
        # tempo_curve[time_kernel] = selected_bpm
        sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * selected_freq - selected_phase))
        estimated_beat_pulse[time_kernel] += sinusoidal_kernel

    tempo_curve = tempo_curve[left_padding:padded_curve_length-right_padding]
    # global_bpm = np.average(tempo_curve)
    
    estimated_beat_pulse = estimated_beat_pulse[left_padding:padded_curve_length-right_padding]
    estimated_beat_pulse[estimated_beat_pulse < 0] = 0

    json_data = {"estimated_beat_pulse": estimated_beat_pulse,
                 "tempo_curve": tempo_curve,
                 "tempo_curve_time_axis": tempo_curve_taxis,
                #  "global_tempo_bpm": global_bpm,
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








def compute_predominant_local_pulse(tempogram, sampling_rate, novelty_length, window_length, hop_size, tempi):
    """Compute windowed sinusoid with optimal phase

    Notebook: C6/C6S3_PredominantLocalPulse.ipynb

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
    padded_novelty_length = novelty_length + left_padding + right_padding
    predominant_local_pulse = np.zeros(padded_novelty_length)
    num_frames = tempogram.shape[1]
    magnitude_tempogram = np.abs(tempogram)
    
    tempo_curve = np.zeros(padded_novelty_length)
    
    for frame_idx in range(num_frames):
        # select peak frequency for a time window
        peak_tempo_idx = np.argmax(magnitude_tempogram[:, frame_idx])
        peak_tempo_bpm = tempi[peak_tempo_idx]
        frequency = (peak_tempo_bpm / 60) / sampling_rate
        
        # get the complex value for that peak frequency and time window
        complex_value = tempogram[peak_tempo_idx, frame_idx]
        phase = - np.angle(complex_value) / (2 * np.pi)
        
        
        start_index = frame_idx * hop_size
        end_index = start_index + window_length
        time_kernel = np.arange(start_index, end_index)
        
        sinusoidal_kernel = hann_window * np.cos(2 * np.pi * (time_kernel * frequency - phase))
        predominant_local_pulse[time_kernel] += sinusoidal_kernel
        
        tempo_curve[time_kernel] = peak_tempo_bpm
        
    predominant_local_pulse = predominant_local_pulse[left_padding:padded_novelty_length-right_padding]
    predominant_local_pulse[predominant_local_pulse < 0] = 0
    
    return predominant_local_pulse, tempo_curve


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


def filter_onsets_by_distance(xyz_ab_minima, xyz_ab, distance_threshold=0.1, time_threshold=0, fps=60):
    
    # xyz_ab_minima: minima from the velocity data, xyz_ab: velocity data
    filtered_onsets = []
    
    # Iterate through the onsets
    for i in range(len(xyz_ab_minima) - 1):
        onset_current = xyz_ab_minima[i]
        onset_next = xyz_ab_minima[i + 1]
        
        # Calculate the distance between the two onsets (in terms of velocity)
        distance = np.sum(np.abs(xyz_ab[onset_current:onset_next])) / fps
        
        # Compute time difference in frames
        time_diff = (onset_next-onset_current)/fps
        
        # Apply the distance threshold
        if distance > distance_threshold and time_diff >= time_threshold:
            # Keep the next onset
            filtered_onsets.append(onset_next)
    
    return np.array(filtered_onsets)

def velocity_based_novelty(velocity_array, order=15):
    
    dir_change_onset_arr = np.array([])
    onset_data_list = []
    for i in range(velocity_array.shape[1]):
        
        maxima_indices = argrelmax(velocity_array[:,i], order=order)[0]
        binary_onset_data = np.zeros(len(velocity_array[:,i]))
        binary_onset_data[maxima_indices] = 1                      # directional change onsets represented by value 1

        onset_data_list.append(binary_onset_data)
    dir_change_onset_arr = np.column_stack(onset_data_list)
    
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