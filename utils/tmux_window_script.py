import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

from compute_tempo_aist import *
from aist_pos1s_extraction import *
from aist_pos1s_EsTempo import *

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        json_data = pickle.load(f)
    return json_data

json_filename = "music_id_tempo.json"
with open(json_filename, "r") as file:
    aist_tempo = json.load(file)

def body_tempo_estimation(a, b, mode, metric, w_sec, h_sec, save_dir):

    segment_keys = [
                "both_hand_x", "both_hand_y", "both_foot_x", "both_foot_y",
                     
                "lefthand_xy", "righthand_xy", "leftfoot_xy", "rightfoot_xy", 
                "bothhand_xy", "bothfoot_xy",
                
                "left_hand_x", "right_hand_x", "left_hand_y", "right_hand_y", 
                "left_foot_x", "right_foot_x", "left_foot_y", "right_foot_y", 
                
                "both_hand_resultant", "both_foot_resultant", "left_hand_resultant", 
                "right_hand_resultant", "left_foot_resultant", "right_foot_resultant"
                ]
    result = { key: {
        "filename": [],
        "dance_genre": [],
        "situation": [],
        "camera_id": [],
        "dancer_id": [],
        "music_id": [],
        "choreo_id": [],
        "music_tempo": [],
        "estimated_bpm_per_window": [],
        "magnitude_per_window": [],
        "bpm_avg": [],
        "bpm_mode": [],
        "bpm_median": [],
    } for key in segment_keys }


    onset_dir = f"./extracted_body_onsets/{metric}/"
    f_path = "./aist_dataset/aist_annotation/keypoints2d"
    aist_filelist = os.listdir(f_path)

    fps = 60
    window_size = int(fps*w_sec)
    hop_size = int(fps*h_sec)
    tempi_range = np.arange(a,b,1)      # tempo step size
    count= 0
    for idx, filename in enumerate(tqdm(aist_filelist)):
        
        file_info = filename.split("_")
        dance_genre = file_info[0] 
        situation = file_info[1] 
        camera_id = file_info[2] 
        dancer_id = file_info[3]
        music_id = file_info[4]
        choreo_id = file_info[5].strip(".pkl")
        
        test_path = os.path.join(onset_dir, "ax0", f"left_wrist_{mode}_{filename}")
        isExist = os.path.exists(test_path)
        if not isExist:
            continue
                                
        left_hand_x  = load_pickle(os.path.join(onset_dir, "ax0", f"left_wrist_{mode}_{filename}"))
        left_hand_y  = load_pickle(os.path.join(onset_dir, "ax1", f"left_wrist_{mode}_{filename}"))
        
        right_hand_x = load_pickle(os.path.join(onset_dir, "ax0", f"right_wrist_{mode}_{filename}"))
        right_hand_y = load_pickle(os.path.join(onset_dir, "ax1", f"right_wrist_{mode}_{filename}"))
        
        left_foot_x  = load_pickle(os.path.join(onset_dir, "ax0", f"left_ankle_{mode}_{filename}"))
        left_foot_y  = load_pickle(os.path.join(onset_dir, "ax1", f"left_ankle_{mode}_{filename}"))
        
        right_foot_x = load_pickle(os.path.join(onset_dir, "ax0", f"right_ankle_{mode}_{filename}"))
        right_foot_y = load_pickle(os.path.join(onset_dir, "ax1", f"right_ankle_{mode}_{filename}"))
        
        novelty_length = left_hand_x['raw_signal'].shape[0]
        
        key = 'sensor_onsets'
        thres = 0.2
        
        bothhand_x = filter_dir_onsets_by_threshold((left_hand_x[key] + right_hand_x[key]), threshold_s= thres, fps=fps)
        bothhand_y = filter_dir_onsets_by_threshold((left_hand_y[key] + right_hand_y[key]), threshold_s= thres, fps=fps)

        bothfoot_x = filter_dir_onsets_by_threshold((left_foot_x[key] + right_foot_x[key]), threshold_s= thres, fps=fps)
        bothfoot_y = filter_dir_onsets_by_threshold((left_foot_y[key] + right_foot_y[key]), threshold_s= thres, fps=fps)
        
        lefthand_xy = filter_dir_onsets_by_threshold((left_hand_x[key] + left_hand_y[key]), threshold_s= thres, fps=fps)
        righthand_xy = filter_dir_onsets_by_threshold((right_hand_x[key] + right_hand_y[key]), threshold_s= thres, fps=fps)

        leftfoot_xy = filter_dir_onsets_by_threshold((left_foot_x[key] + left_foot_y[key]), threshold_s= thres, fps=fps)
        rightfoot_xy = filter_dir_onsets_by_threshold((right_foot_x[key] + right_foot_y[key]), threshold_s= thres, fps=fps)
        
        bothhand_xy = filter_dir_onsets_by_threshold((lefthand_xy + righthand_xy), threshold_s= thres, fps=fps)
        bothfoot_xy = filter_dir_onsets_by_threshold((leftfoot_xy + rightfoot_xy), threshold_s= thres, fps=fps)
        
        # Resultant part
        key1 = 'resultant_onsets'
        left_hand_resultant  = load_pickle(os.path.join(onset_dir, "resultant", f"left_wrist_{mode}_{filename}"))
        right_hand_resultant  = load_pickle(os.path.join(onset_dir, "resultant", f"right_wrist_{mode}_{filename}"))

        left_foot_resultant = load_pickle(os.path.join(onset_dir, "resultant", f"left_ankle_{mode}_{filename}"))
        right_foot_resultant = load_pickle(os.path.join(onset_dir, "resultant", f"right_ankle_{mode}_{filename}"))
        
        both_hand_resultant = filter_dir_onsets_by_threshold((left_hand_resultant[key1] + right_hand_resultant[key1]), threshold_s= thres, fps=fps)
        both_foot_resultant = filter_dir_onsets_by_threshold((left_foot_resultant[key1] + right_foot_resultant[key1]), threshold_s= thres, fps=fps)
        
        
        segment_ax = {
                    "both_hand_x": bothhand_x, "both_hand_y": bothhand_y, "both_foot_x": bothfoot_x, "both_foot_y": bothfoot_y,
                    
                    "lefthand_xy": lefthand_xy, "righthand_xy": righthand_xy, "leftfoot_xy": leftfoot_xy, "rightfoot_xy": rightfoot_xy,
                    "bothhand_xy": bothhand_xy, "bothfoot_xy": bothfoot_xy,
                    
                    "left_hand_x": left_hand_x[key], "right_hand_x": right_hand_x[key], 
                    "left_hand_y": left_hand_y[key], "right_hand_y": right_hand_y[key],
                    
                    "left_foot_x": left_foot_x[key], "right_foot_x": right_foot_x[key],
                    "left_foot_y": left_foot_y[key], "right_foot_y": right_foot_y[key],
                    
                    "both_hand_resultant": both_hand_resultant, "both_foot_resultant": both_foot_resultant,                         
                    "left_hand_resultant": left_hand_resultant[key1], "right_hand_resultant": right_hand_resultant[key1],
                    "left_foot_resultant": left_foot_resultant[key1], "right_foot_resultant": right_foot_resultant[key1],
                    }
        
        for seg_key, seg in segment_ax.items():
            
            sensor_onsets = binary_to_peak(seg, peak_duration=0.05)
            
            tempogram_ab, tempogram_raw, _, _ = compute_tempogram(sensor_onsets, fps, 
                                                                            window_length=window_size, hop_size=hop_size, tempi=tempi_range)
            

            tempo_data_maxmethod = dance_beat_tempo_estimation_maxmethod(tempogram_ab, tempogram_raw, fps, 
                                                            novelty_length, window_size, hop_size, tempi_range)
        

            estimated_bpm_per_window = tempo_data_maxmethod["bpm_arr"]
            magnitude_per_window = tempo_data_maxmethod["mag_arr"]
            
            tempo_avg = np.round(np.average(estimated_bpm_per_window), 2)     # mean
            tempo_mode = stats.mode(estimated_bpm_per_window.flatten())[0]        # 
            tempo_median = np.median(estimated_bpm_per_window.flatten())

            # Append results for the current segment
            result[seg_key]["filename"].append(filename.strip(".pkl"))
            result[seg_key]["dance_genre"].append(dance_genre)
            result[seg_key]["situation"].append(situation)
            result[seg_key]["camera_id"].append(camera_id)
            result[seg_key]["dancer_id"].append(dancer_id)
            result[seg_key]["music_id"].append(music_id)
            result[seg_key]["choreo_id"].append(choreo_id)
            result[seg_key]["music_tempo"].append(aist_tempo[music_id])
            result[seg_key]["estimated_bpm_per_window"].append(estimated_bpm_per_window)
            result[seg_key]["magnitude_per_window"].append(magnitude_per_window)
            result[seg_key]["bpm_avg"].append(tempo_avg)
            result[seg_key]["bpm_mode"].append(tempo_mode)
            result[seg_key]["bpm_median"].append(tempo_median)
        count +=1
    print("total processed:",count)    
    for seg_key in segment_keys:
        
        fname = f"{seg_key}_{mode}_W{w_sec}_H{h_sec}_{a}_{b}.pkl"
        pickle_filename = os.path.join(save_dir, fname)
        # pickle_filename = f"./saved_results/window_cases/{metric}/{seg_key}_{mode}_W{w_sec}_H{h_sec}_{a}_{b}.pkl"
        df_seg = pd.DataFrame(result[seg_key])
        df_seg.to_pickle(pickle_filename)
        print(f"Saved {fname}")
   
        
config1 = {"a": 60, "b": 140, "mode": ["zero_uni", "zero_bi"], "metric": ["pos", "vel"],
           "win_size_list": np.arange(1,14).tolist()}

configs = [config1]

for cfg in configs:
    a = cfg["a"]
    b = cfg["b"]
    for metric in cfg["metric"]:
        for mode in cfg["mode"]:
            
            for w_sec in cfg['win_size_list']:
                for h_sec in tqdm([w_sec*0.25, w_sec*0.5, w_sec*0.75, w_sec*1], desc="hop size"):
                    save_dir = f"./saved_result/window_cases/{metric}/"
                    print(metric, mode)
                    body_tempo_estimation(a, b, mode, metric, w_sec, h_sec, save_dir)




