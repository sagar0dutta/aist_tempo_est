import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from compute_tempo import *
import librosa

marker_dict = {9: "left_wrist", 10: "right_wrist", 
                15: "left_ankle", 16: "right_ankle", 
                }   # 11: "left_hip",12: "right_hip"

def extract_body_onsets(mode, markerA_id, savepath, h_thres = 0.2, vel_mode = "off"):

    fps = 60
    f_path = "./aist_dataset/aist_annotation/keypoints2d"
    aist_filelist = os.listdir(f_path)

    skipped_list = []
    for idx, filename in enumerate(tqdm(aist_filelist)):
        
        file_path = os.path.join(f_path, filename)
        
        with open(file_path, 'rb') as file:
            motion_data = pickle.load(file)

        markerA_x = motion_data["keypoints2d"][0, :, markerA_id, 0]     # array (n,)
        markerA_y = motion_data["keypoints2d"][0, :, markerA_id, 1]      # array (n,)
        
        if np.all((markerA_x == 0) & (markerA_y == 0)) or np.any(np.isnan(markerA_x) | np.isnan(markerA_y)):
            skipped_list.append(filename)
            continue
        
        markerA_x = detrend_signal_array(markerA_x.reshape(-1, 1), cutoff= 1, fs=60)
        markerA_y = detrend_signal_array(markerA_y.reshape(-1, 1), cutoff= 1, fs=60)
        markerA_pos = np.concatenate((markerA_x, markerA_y), axis=1)  # size (n,2)
        
        resultant = np.sqrt(markerA_x**2 + markerA_y**2)
        # min max -1 to 1
        x_min, x_max = np.min(resultant), np.max(resultant)
        resultant_norm = (
            (resultant - x_min) / (x_max - x_min)
            if x_max != x_min  # Avoid division by zero
            else np.zeros_like(resultant)
        )
        
        for ax in range(2):
            
            # z-score
            mean_x = np.mean(markerA_pos[:, ax])
            std_x = np.std(markerA_pos[:, ax])
            
            markerA_pos_norm = (
            (markerA_pos[:, ax] - mean_x) / std_x if std_x != 0 else np.zeros_like(markerA_pos[:, ax])
            )

            markerA_ax = markerA_pos_norm.reshape(-1,1)
                
            
            bodysegment_onsets_data = extract_dance_onset(markerA_ax, T_filter=0.25, 
                                                            smooth_wlen= 10, pk_order = 15, 
                                                            remove_pk_thres=0.10, height_thres=h_thres,
                                                            mov_avg_winsz = 10, fps = fps,
                                                            vel_mode= vel_mode, mode = mode)

            # Save body segment onsets
            save_to_pickle(savepath, f"ax{ax}/{marker_dict[markerA_id]}_{mode}_{filename}", bodysegment_onsets_data)
        
        resultant_norm = resultant_norm.reshape(-1,1)
        resultant_onsets_data = extract_resultant_dance_onset(resultant_norm, T_filter=0.25, 
                                                            smooth_wlen= 10, pk_order = 30,
                                                            remove_pk_thres=0.10, height_thres=h_thres,
                                                            mov_avg_winsz = 10, fps =60,
                                                            vel_mode= vel_mode)    
    
        save_to_pickle(savepath, f"resultant/{marker_dict[markerA_id]}_{mode}_{filename}", resultant_onsets_data)

def extract_dance_onset(sensor_data, T_filter=0.25, 
                        smooth_wlen= 10, pk_order = 30,
                        remove_pk_thres=0.10, height_thres=0.2,
                        mov_avg_winsz = 10, fps =60,
                        vel_mode="off", mode = "zero_uni"):
    # to used for any combincation of two sensors or two body markers
    sensor_dir_change = None
    sensor_onsets = None
    hop_length = 6
    frame_length = 30

    if mode == 'zero_uni':          # Extract uni-directional change onsets
        
        sensor_abs_pos = smooth_velocity(sensor_data, abs="no", window_length = smooth_wlen, polyorder = 0) # size (n, 3)

        sensor_abs_pos[sensor_abs_pos < 0] = 0
        
        # sensor_abs_pos = np.square(np.diff(sensor_abs_pos, axis=0) )   # velocity
        
        rmse = librosa.feature.rms(y=sensor_abs_pos, frame_length=frame_length, hop_length=hop_length).flatten()
        # rmse_diff = np.zeros_like(rmse)
        # rmse_diff[1:] = np.diff(rmse)
        # energy_novelty = np.max([np.zeros_like(rmse_diff), rmse_diff], axis=0)
        
        rmse_smooth = np.convolve(rmse, np.ones(3)/3, mode='same')  # small moving avg
        rmse_diff = np.diff(rmse_smooth, prepend=rmse_smooth[0])
        energy_novelty = np.maximum(0, rmse_diff)
        
        
        sensor_abs_pos_norm = min_max_normalize_1D(energy_novelty.flatten())     # normalize 0-1
        sensor_dir_change = velocity_based_novelty(sensor_abs_pos_norm.reshape(-1,1), height = height_thres, distance=15)    # size (n, 3)
        
        sensor_onsets = filter_dir_onsets_by_threshold(sensor_dir_change, threshold_s= T_filter, fps=fps)
        sensor_onsets_50ms = binary_to_peak(sensor_onsets, peak_duration=0.05)
  

    elif mode == 'zero_bi':         # Extract bi-directional change onsets
        
        sensor_abs_pos = smooth_velocity(sensor_data, abs="yes", window_length = smooth_wlen, polyorder = 0) # size (n, 3)

        # sensor_abs_pos = np.square(np.diff(sensor_abs_pos, axis=0))   # velocity
        
        rmse = librosa.feature.rms(y=sensor_abs_pos, frame_length=frame_length, hop_length=hop_length).flatten()
        # rmse_diff = np.zeros_like(rmse)
        # rmse_diff[1:] = np.diff(rmse)
        # energy_novelty = np.max([np.zeros_like(rmse_diff), rmse_diff], axis=0)
        
        rmse_smooth = np.convolve(rmse, np.ones(3)/3, mode='same')  # small moving avg
        rmse_diff = np.diff(rmse_smooth, prepend=rmse_smooth[0])
        energy_novelty = np.maximum(0, rmse_diff)
        
    
        sensor_abs_pos_norm = min_max_normalize_1D(energy_novelty.flatten())
        sensor_dir_change = velocity_based_novelty(sensor_abs_pos_norm.reshape(-1,1), height = height_thres, distance=15)    # size (n, 3)
        
        sensor_onsets = filter_dir_onsets_by_threshold(sensor_dir_change, threshold_s= T_filter, fps=fps)
        sensor_onsets_50ms = binary_to_peak(sensor_onsets, peak_duration=0.05)

    json_data = {
        "raw_signal": sensor_data,
        "sensor_abs": sensor_abs_pos,   # array
        'sensor_abs_pos_norm': sensor_abs_pos_norm,
        "sensor_dir_change_onsets": sensor_dir_change,  # array
        "sensor_onsets": sensor_onsets,     # array
        "sensor_onsets_50ms": sensor_onsets_50ms,     # array
        
        
    }
    return json_data

def extract_resultant_dance_onset(resultant_data, T_filter=0.20, 
                        smooth_wlen= 10, pk_order = 30,
                        remove_pk_thres=0.10, height_thres=0.2,
                        mov_avg_winsz = 10, fps =60,
                        vel_mode="off"):

    hop_length = 15
    frame_length = 30
    
    ########### For resultant of x and y  ##################
    resultant_smooth = smooth_velocity(resultant_data, abs="no", window_length = smooth_wlen, polyorder = 0) # size (n, 3)
    # if vel_mode == "on":
        # resultant_smooth = np.diff(resultant_smooth, axis=0)    # velocity
    
    # new update: peak filtering and moving average
    # resultant_filtered = remove_low_peaks(resultant_smooth.flatten(), remove_pk_thres, rel_height)
    rmse = librosa.feature.rms(y=resultant_smooth, frame_length=frame_length, hop_length=hop_length).flatten()
    # rmse_diff = np.zeros_like(rmse)
    # rmse_diff[1:] = np.diff(rmse)
    # energy_novelty = np.max([np.zeros_like(rmse_diff), rmse_diff], axis=0)
    
    rmse_smooth = np.convolve(rmse, np.ones(3)/3, mode='same')  # small moving avg
    rmse_diff = np.diff(rmse_smooth, prepend=rmse_smooth[0])
    energy_novelty = np.maximum(0, rmse_diff)
    
    
    
    resultant_filtered = moving_average(energy_novelty.flatten(), mov_avg_winsz)
    
    resultant_dir_change = velocity_based_novelty(resultant_filtered.reshape(-1,1), height = height_thres, distance=15)    # size (n, 3)
    resultant_onsets = filter_dir_onsets_by_threshold(resultant_dir_change, threshold_s= T_filter, fps=fps)
    resultant_onsets_50ms = binary_to_peak(resultant_onsets, peak_duration=0.05)

    json_data = {
        "resultant_signal": resultant_data,
        "resultant_smooth": resultant_smooth,   # array
        "resultant_filtered": resultant_filtered,   # array
        "resultant_dir_change": resultant_dir_change,  # array
        "resultant_onsets": resultant_onsets,     # array
        "resultant_onsets_50ms": resultant_onsets_50ms,     # array 
    }
    return json_data

def min_max_normalize_1D(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_to_pickle(savepath, filename, json_tempodata):
    filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(json_tempodata, f)