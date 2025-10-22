import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from .compute_tempo import *
from .anchor_io import *


# -------------------------------------------------------------
# 1. File info extraction
# -------------------------------------------------------------
def parse_filename(filename):
    """Extract metadata from filename."""
    file_info = filename.split("_")
    return {
        "dance_genre": file_info[0],
        "situation": file_info[1],
        "camera_id": file_info[2],
        "dancer_id": file_info[3],
        "music_id": file_info[4],
        "choreo_id": file_info[5].replace(".pkl", "")
    }


# -------------------------------------------------------------
# 2. Data loading helpers
# -------------------------------------------------------------
def load_marker_data(paths, key="sensor_onsets"):
    """Load hand/foot marker pickles and return dictionary of onsets."""
    def lp(p): return load_pickle(p)[key]
    
    return {
        "left_hand_x": lp(paths["markers"]["left_wrist"]["ax0"]),
        "left_hand_y": lp(paths["markers"]["left_wrist"]["ax1"]),
        "right_hand_x": lp(paths["markers"]["right_wrist"]["ax0"]),
        "right_hand_y": lp(paths["markers"]["right_wrist"]["ax1"]),
        "left_foot_x": lp(paths["markers"]["left_ankle"]["ax0"]),
        "left_foot_y": lp(paths["markers"]["left_ankle"]["ax1"]),
        "right_foot_x": lp(paths["markers"]["right_ankle"]["ax0"]),
        "right_foot_y": lp(paths["markers"]["right_ankle"]["ax1"]),
        "torso_y": lp(paths["com"]["com_torso"]["ax1"]),
    }


# -------------------------------------------------------------
# 3. Combine and filter onset sequences
# -------------------------------------------------------------
def compute_combinations(data, fps, thres):
    """Compute all combinations of hand/foot onsets."""
    f = filter_dir_onsets_by_threshold
    k = "both"

    bothhand_x = f(data["left_hand_x"] + data["right_hand_x"], threshold_s=thres, fps=fps)
    bothhand_y = f(data["left_hand_y"] + data["right_hand_y"], threshold_s=thres, fps=fps)
    bothfoot_x = f(data["left_foot_x"] + data["right_foot_x"], threshold_s=thres, fps=fps)
    bothfoot_y = f(data["left_foot_y"] + data["right_foot_y"], threshold_s=thres, fps=fps)

    lefthand_xy = f(data["left_hand_x"] + data["left_hand_y"], threshold_s=thres, fps=fps)
    righthand_xy = f(data["right_hand_x"] + data["right_hand_y"], threshold_s=thres, fps=fps)
    leftfoot_xy = f(data["left_foot_x"] + data["left_foot_y"], threshold_s=thres, fps=fps)
    rightfoot_xy = f(data["right_foot_x"] + data["right_foot_y"], threshold_s=thres, fps=fps)

    return {
        **data,
        "both_hand_x": bothhand_x, "both_hand_y": bothhand_y,
        "both_foot_x": bothfoot_x, "both_foot_y": bothfoot_y,
        "lefthand_xy": lefthand_xy, "righthand_xy": righthand_xy,
        "leftfoot_xy": leftfoot_xy, "rightfoot_xy": rightfoot_xy,
        "bothhand_x_bothfoot_x": f(bothhand_x + bothfoot_x, threshold_s=thres, fps=fps),
        "bothhand_y_bothfoot_y": f(bothhand_y + bothfoot_y, threshold_s=thres, fps=fps),
        "lefthand_xy_righthand_xy": f(lefthand_xy + righthand_xy, threshold_s=thres, fps=fps),
        "leftfoot_xy_rightfoot_xy": f(leftfoot_xy + rightfoot_xy, threshold_s=thres, fps=fps),
        "bothhand_x_bothhand_y": f(bothhand_x + bothhand_y, threshold_s=thres, fps=fps),
        "bothfoot_x_bothfoot_y": f(bothfoot_x + bothfoot_y, threshold_s=thres, fps=fps),
    }


# -------------------------------------------------------------
# 4. Compute resultant and append
# -------------------------------------------------------------
def load_resultant(paths, thres, fps):
    """Load and merge resultant signals."""
    key = "resultant_onsets"
    f = filter_dir_onsets_by_threshold
    
    def lr(p): return load_pickle(p)[key]
    left_hand_r = lr(paths["markers"]["left_wrist"]["resultant"])
    right_hand_r = lr(paths["markers"]["right_wrist"]["resultant"])
    left_foot_r = lr(paths["markers"]["left_ankle"]["resultant"])
    right_foot_r = lr(paths["markers"]["right_ankle"]["resultant"])

    both_hand_r = f(left_hand_r + right_hand_r, threshold_s=thres, fps=fps)
    both_foot_r = f(left_foot_r + right_foot_r, threshold_s=thres, fps=fps)

    return {
        "left_hand_resultant": left_hand_r,
        "right_hand_resultant": right_hand_r,
        "left_foot_resultant": left_foot_r,
        "right_foot_resultant": right_foot_r,
        "both_hand_resultant": both_hand_r,
        "both_foot_resultant": both_foot_r,
    }


# -------------------------------------------------------------
# 5. Tempo computation per segment
# -------------------------------------------------------------
def compute_tempo_for_segments(segment_ax, fps, window_size, hop_size, tempi_range, novelty_length):
    """Compute tempogram and tempo statistics for each segment."""
    tempo_data = {}

    for seg_key, seg in segment_ax.items():
        sensor_onsets = binary_to_peak(seg, peak_duration=0.1)
        tempogram_ab, tempogram_raw, _, _ = compute_tempogram(sensor_onsets, fps, window_size, hop_size, tempi_range)
        tempo_info = dance_tempo_estimation_single(tempogram_ab[0], tempogram_raw[0], fps, novelty_length, window_size, hop_size, tempi_range)

        bpm_arr = tempo_info["bpm_arr"].flatten()
        tempo_data[seg_key] = {
            **tempo_info,
            "bpm_avg": np.round(np.average(bpm_arr), 2),
            "bpm_mode": stats.mode(bpm_arr)[0],
            "bpm_median": np.median(bpm_arr)
        }

    return tempo_data