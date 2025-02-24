import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from compute_tempo_aist import *

def aist_pos1s(a,b, mode, markerA_id, csv_filename):

    result = {
        "filename": [],
        "dance_genre": [],
        "situation": [],
        "camera_id": [],
        "dancer_id": [],
        "music_id": [],
        "choreo_id": [],
        "music_tempo": [],

        "X_a": [],
        "Y_a": [],

        "bpm_mode": [],
        "bpm_median": [],
        "mode_x": [],
        "mode_y": [],

        "median_x": [],
        "median_y": [],

    }

    json_filename = "music_id_tempo.json"
    with open(json_filename, "r") as file:
        aist_tempo = json.load(file)

    f_path = "./aist_dataset/aist_annotation/keypoints2d"
    aist_filelist = os.listdir(f_path)

    mocap_fps = 60
    # a = 70; b =145
    tempi_range = np.arange(a,b,1)   # good: 70,145 

    # marker_dict = {9: "left_wrist", 10: "right_wrist", 
    #             15: "left_ankle", 16: "right_ankle", 
    #             11: "left_hip",12: "right_hip"}


    # sub_dir = "foot"
    # mode = "zero_bi"
    # markerA_id = 16          # 9: "left_wrist", 10: "right_wrist", 15: "left_ankle", 16: "right_ankle"
    # csv_filename = f"./results/aist_pos1s/z-score/tempo_{a}_{b}/{sub_dir}/{marker_dict[markerA_id]}_{mode}_{a}_{b}.csv"        

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
        
        

        if np.all((markerA_x == 0) & (markerA_y == 0)) or np.any(np.isnan(markerA_x) | np.isnan(markerA_y)):
            continue
        
        markerA_x = detrend_signal_array(markerA_x.reshape(-1, 1), cutoff= 1, fs=60)
        markerA_y = detrend_signal_array(markerA_y.reshape(-1, 1), cutoff= 1, fs=60)
        markerA_pos = np.concatenate((markerA_x, markerA_y), axis=1)  # size (n,2)
        
        duration = int(len(markerA_x)/60)
        w_sec = int(duration)
        h_sec = int(w_sec/4)
        window_size = int(mocap_fps*w_sec)
        hop_size = int(mocap_fps*h_sec)
        
        bpm_axes = []
        for ax in range(2):
            # min max -1 to 1
            # x_min, x_max = np.min(markerA_pos[:, ax]), np.max(markerA_pos[:, ax])
            # markerA_pos_norm = (
            #     2 * (markerA_pos[:, ax] - x_min) / (x_max - x_min) - 1
            #     if x_max != x_min  # Avoid division by zero
            #     else np.zeros_like(markerA_pos[:, ax])
            # )
            
            # z-score
            mean_x = np.mean(markerA_pos[:, ax])
            std_x = np.std(markerA_pos[:, ax])
            
            markerA_pos_norm = (
            (markerA_pos[:, ax] - mean_x) / std_x if std_x != 0 else np.zeros_like(markerA_pos[:, ax])
            )

            markerA_ax = markerA_pos_norm.reshape(-1,1)
                
            tempo_json_one_sensor = main_one_sensor_peraxis(markerA_ax,
                                                            mocap_fps, window_size, hop_size, tempi_range, 
                                                            T_filter=0.25, smooth_wlen= 10, pk_order = 15 ,
                                                            vel_mode="on",mode= mode)

            
            sensor_onsets = tempo_json_one_sensor["sensor_onsets"]
            tempo_data_maxmethod = tempo_json_one_sensor["tempo_data_maxmethod"]
            bpmA_arr = tempo_data_maxmethod["bpm_arr"]
            tempo_A = np.round(np.average(bpmA_arr), 2)
            bpm_axes.append(bpmA_arr)
            
            mode_x = stats.mode(bpmA_arr.flatten())[0]
            mode_y = stats.mode(bpmA_arr.flatten())[0]

            median_x = np.median(bpmA_arr.flatten())
            median_y = np.median(bpmA_arr.flatten())

            
            if ax == 0:
                result["filename"].append(filename.strip(".pkl"))
                result["dance_genre"].append(dance_genre)
                result["situation"].append(situation)
                result["camera_id"].append(camera_id)
                result["dancer_id"].append(dancer_id)
                result["music_id"].append(music_id)
                result["choreo_id"].append(choreo_id)
                result["music_tempo"].append(aist_tempo[music_id])
                
                result["mode_x"].append(mode_x)
                result["mode_y"].append(mode_y)

                
                result["median_x"].append(median_x)
                result["median_y"].append(median_y)

                
                result["X_a"].append(tempo_A)

            elif ax == 1:
                result["Y_a"].append(tempo_A)

        bpm_axes_arr = np.column_stack(bpm_axes)    # n by 3 array
        bpm_mode = stats.mode(bpm_axes_arr.flatten())[0]
        bpm_median = np.median(bpm_axes_arr.flatten())
        result["bpm_mode"].append(bpm_mode)
        result["bpm_median"].append(bpm_median)    
        
        
    results_df = pd.DataFrame(result)
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


