import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from compute_tempo_aist import *

marker_dict = {9: "left_wrist", 10: "right_wrist", 
                15: "left_ankle", 16: "right_ankle", 
                }   # 11: "left_hip",12: "right_hip"

def aist_pos2s(a,b, mode, markerA_id, markerB_id, csv_filename, savepath, w_sec = 5, vel_mode = "off"):

    result = {
        "filename": [],
        "dance_genre": [],
        "situation": [],
        "camera_id": [],
        "dancer_id": [],
        "music_id": [],
        "choreo_id": [],
        "music_tempo": [],

        "estimated_bpm_per_window_x":[],
        "estimated_bpm_per_window_y":[],
        
        "bpm_avg_x": [],
        "bpm_avg_y": [],
        "bpm_avg_xy":[],
        
        "bpm_mode_x": [],
        "bpm_mode_y": [],
        "bpm_mode_xy": [],

        "bpm_median_x": [],
        "bpm_median_y": [],
        "bpm_median_xy": [],
    }

    json_filename = "music_id_tempo.json"
    with open(json_filename, "r") as file:
        aist_tempo = json.load(file)

    f_path = "./aist_dataset/aist_annotation/keypoints2d"
    aist_filelist = os.listdir(f_path)

    mocap_fps = 60
    tempi_range = np.arange(a,b,1)   # good: 70,145 

    for idx, filename in enumerate(tqdm(aist_filelist)):
        
        file_path = os.path.join(f_path, filename)
        file_info = filename.split("_")
        dance_genre = file_info[0] 
        situation = file_info[1] 
        camera_id = file_info[2] 
        dancer_id = file_info[3]
        music_id = file_info[4]
        choreo_id = file_info[5].strip(".pkl")
        
        with open(file_path, 'rb') as file:
            motion_data = pickle.load(file)

        markerA_x = motion_data["keypoints2d"][0, :, markerA_id, 0]     # array (n,)
        markerA_y = motion_data["keypoints2d"][0, :, markerA_id, 1]      # array (n,)
        
        markerB_x = motion_data["keypoints2d"][0, :, markerB_id, 0]     # array (n,)
        markerB_y = motion_data["keypoints2d"][0, :, markerB_id, 1]      # array (n,)
        

        if np.all((markerA_x == 0) & (markerA_y == 0)) or np.any(np.isnan(markerA_x) | np.isnan(markerA_y)):
            continue
        
        markerA_x = detrend_signal_array(markerA_x.reshape(-1, 1), cutoff= 1, fs=60)
        markerA_y = detrend_signal_array(markerA_y.reshape(-1, 1), cutoff= 1, fs=60)
        markerA_pos = np.concatenate((markerA_x, markerA_y), axis=1)  # size (n,2)
        
        markerB_x = detrend_signal_array(markerB_x.reshape(-1, 1), cutoff= 1, fs=60)
        markerB_y = detrend_signal_array(markerB_y.reshape(-1, 1), cutoff= 1, fs=60)
        markerB_pos = np.concatenate((markerB_x, markerB_y), axis=1)  # size (n,2)
        
        
        
        # duration = int(len(markerA_x)/60)
        # w_sec = int(duration)
        h_sec = int(w_sec/4)
        window_size = int(mocap_fps*w_sec)
        hop_size = int(mocap_fps*h_sec)
        
        bpm_axes = []
        for ax in range(2):
            
            # min max -1 to 1
            # posA_min, posA_max = np.min(markerA_pos[:, ax]), np.max(markerA_pos[:, ax])
            # sensorA_position_norm = (
            #         2*(markerA_pos[:, ax] - posA_min) / (posA_max - posA_min) - 1
            #         if posA_max != posA_min 
            #         else np.zeros_like(markerA_pos[:, ax])
            #     )

            # posB_min, posB_max = np.min(markerB_pos[:, ax]), np.max(markerB_pos[:, ax])
            # sensorB_position_norm = (
            #         2*(markerB_pos[:, ax] - posB_min) / (posB_max - posB_min) - 1
            #         if posB_max != posB_min
            #         else np.zeros_like(markerB_pos[:, ax])
            #     )
            
            # z-score normalization for markerA
            posA_mean, posA_std = np.mean(markerA_pos[:, ax]), np.std(markerA_pos[:, ax])
            sensorA_position_norm = (
                (markerA_pos[:, ax] - posA_mean) / posA_std if posA_std != 0 else np.zeros_like(markerA_pos[:, ax])
            )
            
            # Z-score normalization for markerB
            posB_mean, posB_std = np.mean(markerB_pos[:, ax]), np.std(markerB_pos[:, ax])
            sensorB_position_norm = (
                (markerB_pos[:, ax] - posB_mean) / posB_std if posB_std != 0 else np.zeros_like(markerB_pos[:, ax])
            )
            
            
            tempo_json = main_two_sensor(sensorA_position_norm.reshape(-1,1), sensorB_position_norm.reshape(-1,1),  
                                        mocap_fps, window_size, hop_size, tempi_range,
                                        T_filter= 0.25, smooth_wlen= 10, pk_order = 15, vel_mode = vel_mode, mode=mode)
        
            tempo_data_maxmethod = tempo_json["tempo_data_maxmethod"]
            bpmA_arr = tempo_data_maxmethod["bpm_arr"]
            tempo_avg_ax = np.round(np.average(bpmA_arr), 2)     # mean
            bpm_axes.append(bpmA_arr)
            
            mode_ax = stats.mode(bpmA_arr.flatten())[0]        # 
            median_ax = np.median(bpmA_arr.flatten())

            
            if ax == 0:
                result["filename"].append(filename.strip(".pkl"))
                result["dance_genre"].append(dance_genre)
                result["situation"].append(situation)
                result["camera_id"].append(camera_id)
                result["dancer_id"].append(dancer_id)
                result["music_id"].append(music_id)
                result["choreo_id"].append(choreo_id)
                result["music_tempo"].append(aist_tempo[music_id])
                
                result["estimated_bpm_per_window_x"].append(bpmA_arr)
                result["bpm_avg_x"].append(tempo_avg_ax)
                result["bpm_mode_x"].append(mode_ax)
                result["bpm_median_x"].append(median_ax)

            elif ax == 1:
                result["estimated_bpm_per_window_y"].append(bpmA_arr)
                result["bpm_avg_y"].append(tempo_avg_ax)
                result["bpm_mode_y"].append(mode_ax)
                result["bpm_median_y"].append(median_ax)

            # Save tenpo data
            save_to_pickle(savepath, f"ax{ax}/{marker_dict[markerA_id]}_{mode}_{filename}", tempo_json)

        bpm_axes_arr = np.column_stack(bpm_axes)    # n by 3 array
        bpm_mode = stats.mode(bpm_axes_arr.flatten())[0]
        bpm_median = np.median(bpm_axes_arr.flatten())
        bpm_mean = np.mean(bpm_axes_arr.flatten())
        
        result["bpm_avg_xy"].append(bpm_mean)
        result["bpm_mode_xy"].append(bpm_mode)
        result["bpm_median_xy"].append(bpm_median)    
        
        
    results_df = pd.DataFrame(result)
    results_df.to_csv(csv_filename, index=False)   # saves final bpms
    print(f"Results saved to {csv_filename}")

    
def save_to_pickle(savepath, filename, json_tempodata):
    """Save json_tempodata as a Pickle (.pkl) file."""
    filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(json_tempodata, f)

def load_from_pickle(savepath, filename):
    """Load json_tempodata from a Pickle (.pkl) file."""
    filepath = os.path.join(savepath, filename)
    with open(filepath, "rb") as f:
        json_tempodata = pickle.load(f)
    return json_tempodata
