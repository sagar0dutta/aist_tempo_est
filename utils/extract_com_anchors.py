import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from .compute_tempo import *
from .rms_extract_dance_onsets import detect_energy_anchor

marker_dict = {    
0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear",4: "right_ear",5: "left_shoulder",
6: "right_shoulder",7: "left_elbow",8: "right_elbow",9: "left_wrist",10: "right_wrist",
11: "left_hip",12: "right_hip",13: "left_knee",14: "right_knee",15: "left_ankle",16: "right_ankle",}  


def compute_center_of_mass(f_path, save_dirs):

    aist_filelist = os.listdir(f_path)
    skipped_list = []

    for idx, filename in enumerate(tqdm(aist_filelist)):
        file_path = os.path.join(f_path, filename)
        
        try:
            with open(file_path, 'rb') as file:
                motion_data = pickle.load(file)

            # keypoints2d shape: (num_persons, num_frames, num_joints, 3)
            keypoints2d = motion_data["keypoints2d"][0, :, :, :2]
            
            # Compute CoM trajectory
            com_hips, com_shoulders, com_torso = compute_com_variants(keypoints2d)

            # Save each variant in its own directory
            save_paths = {
                "hips": os.path.join(save_dirs["hips"], filename),
                "shoulders": os.path.join(save_dirs["shoulders"], filename),
                "torso": os.path.join(save_dirs["torso"], filename),
            }

            # Save each variant in its own directory
            np.save(os.path.join(save_dirs["hips"], filename.replace(".pkl", ".npy")), com_hips)
            np.save(os.path.join(save_dirs["shoulders"], filename.replace(".pkl", ".npy")), com_shoulders)
            np.save(os.path.join(save_dirs["torso"], filename.replace(".pkl", ".npy")), com_torso)

        except Exception as e:
            skipped_list.append(filename)
            print(f"Skipped {filename}: {e}")

    print(f"Skipped files: {len(skipped_list)}")
    
    
def compute_com_variants(joints_2d):
    """
    Compute 2D center-of-mass variants from COCO keypoints.
    
    Parameters
    ----------
    joints_2d : np.ndarray
        Array of shape (num_persons, num_frames, num_joints, 2)
        from AIST++ keypoints2d data.
    
    Returns
    -------
    com_hips, com_shoulders, com_torso : np.ndarray
        Each of shape (num_frames, 2)
    """
    # COCO joint indices
    LHip, RHip, LShoulder, RShoulder = 11, 12, 5, 6

    # Take first person (index 0)
    j2d = joints_2d  # shape: (num_frames, num_joints, 2)

    # CoM variants
    com_hips = (j2d[:, LHip, :] + j2d[:, RHip, :]) / 2.0
    com_shoulders = (j2d[:, LShoulder, :] + j2d[:, RShoulder, :]) / 2.0
    com_torso = (com_hips + com_shoulders) / 2.0  # mean of hips + shoulders

    return com_hips, com_shoulders, com_torso


def extract_CoM_anchor(com_dirs,mode, savepath, fps =60, h_thres = 0.2, vel_mode="off"):

    # fps = 60
    
    # com_dirs = {
    # "hips": "./extracted_body_onsets/com2d_hips",
    # "shoulders": "./extracted_body_onsets/com2d_shoulders",
    # "torso": "./extracted_body_onsets/com2d_torso"
    # }
    
    aist_filelist = sorted(os.listdir(com_dirs["hips"]))

    for idx, filename in enumerate(tqdm(aist_filelist)):

        com_hips = np.load(os.path.join(com_dirs["hips"], filename))
        com_shoulders = np.load(os.path.join(com_dirs["shoulders"], filename))
        com_torso = np.load(os.path.join(com_dirs["torso"], filename))

        detrend_hip = detrend_signal_array(com_hips, cutoff= 0.5)
        detrend_shoulders = detrend_signal_array(com_shoulders, cutoff= 0.5)
        detrend_torso = detrend_signal_array(com_torso, cutoff= 0.5)
        
        
        for CoM_data, CoM_name in zip([detrend_hip, detrend_shoulders, detrend_torso],
                                      ["hips", "shoulders", "torso"]):
            
            for ax in range(2):
                
                # z-score
                mean_x = np.mean(CoM_data[:, ax])
                std_x = np.std(CoM_data[:, ax])
                
                CoM_norm = (
                (CoM_data[:, ax] - mean_x) / std_x if std_x != 0 else np.zeros_like(CoM_data[:, ax])
                )

                CoM_ax = CoM_norm.reshape(-1,1)
                
                anchor_data = detect_anchor(CoM_ax, T_filter=0.25, 
                                            smooth_wlen= 10, height_thres=h_thres,
                                            fps = fps,
                                            vel_mode= vel_mode, mode = mode)
                
                

                # Save body segment onsets
                save_to_pickle(savepath, f"com_{CoM_name}/ax{ax}/{mode}_{filename.replace("npy", "pkl")}", anchor_data)
        
def extract_CoM_energy_anchor(com_dirs,mode, savepath, fps =60, h_thres = 0.2):

    # fps = 60
    
    # com_dirs = {
    # "hips": "./extracted_body_onsets/com2d_hips",
    # "shoulders": "./extracted_body_onsets/com2d_shoulders",
    # "torso": "./extracted_body_onsets/com2d_torso"
    # }
    
    aist_filelist = sorted(os.listdir(com_dirs["hips"]))

    for idx, filename in enumerate(tqdm(aist_filelist)):

        com_hips = np.load(os.path.join(com_dirs["hips"], filename))
        com_shoulders = np.load(os.path.join(com_dirs["shoulders"], filename))
        com_torso = np.load(os.path.join(com_dirs["torso"], filename))

        detrend_hip = detrend_signal_array(com_hips, cutoff= 0.5)
        detrend_shoulders = detrend_signal_array(com_shoulders, cutoff= 0.5)
        detrend_torso = detrend_signal_array(com_torso, cutoff= 0.5)
        
        
        for CoM_data, CoM_name in zip([detrend_hip, detrend_shoulders, detrend_torso],
                                      ["hips", "shoulders", "torso"]):
            
            for ax in range(2):
                
                # z-score
                mean_x = np.mean(CoM_data[:, ax])
                std_x = np.std(CoM_data[:, ax])
                
                CoM_norm = (
                (CoM_data[:, ax] - mean_x) / std_x if std_x != 0 else np.zeros_like(CoM_data[:, ax])
                )

                CoM_ax = CoM_norm.reshape(-1,1)
                
                        
                energy_anchor_data = detect_energy_anchor(CoM_ax, T_filter=0.25, 
                                            smooth_wlen= 10, height_thres=h_thres,
                                            fps = fps,
                                            mode = mode)
                

                # Save body segment onsets
                save_to_pickle(savepath, f"com_{CoM_name}/ax{ax}/{mode}_{filename.replace("npy", "pkl")}", energy_anchor_data)
    

def detect_anchor(sensor_data, T_filter=0.25, 
                        smooth_wlen= 10, height_thres=0.2,
                        fps =60,
                        vel_mode="off", mode = "uni"):
    # to used for any combincation of two sensors or two body markers
    sensor_dir_change = None
    sensor_onsets = None

    if mode == 'uni':          # Extract uni-directional change onsets
        
        sensor_abs_pos = smooth_velocity(sensor_data, abs="no", window_length = smooth_wlen, polyorder = 0) # size (n, 3)
        if vel_mode == "on":
            sensor_abs_pos = np.diff(sensor_abs_pos, axis=0)    # velocity
        
        sensor_abs_pos[sensor_abs_pos < 0] = 0
        
        sensor_abs_pos_norm = min_max_normalize_1D(sensor_abs_pos.flatten())     # normalize 0-1
        sensor_dir_change = velocity_based_novelty(sensor_abs_pos_norm.reshape(-1,1), height = height_thres, distance=15)    # size (n, 3)
        
        sensor_onsets = filter_dir_onsets_by_threshold(sensor_dir_change, threshold_s= T_filter, fps=fps)
        sensor_onsets_50ms = binary_to_peak(sensor_onsets, peak_duration=0.05)
  

    elif mode == 'bi':         # Extract bi-directional change onsets
        
        sensor_abs_pos = smooth_velocity(sensor_data, abs="yes", window_length = smooth_wlen, polyorder = 0) # size (n, 3)
        if vel_mode == "on":
            sensor_abs_pos = np.diff(sensor_abs_pos, axis=0)   # velocity
        
        sensor_abs_pos_norm = min_max_normalize_1D(sensor_abs_pos.flatten())
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

def min_max_normalize_1D(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_to_pickle(savepath, filename, json_tempodata):
    filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(json_tempodata, f)