import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from .compute_tempo import *
from .anchor_io import *

with open("music_id_tempo.json", "r") as file:
    aist_tempo = json.load(file)


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
        "torso_x": lp(paths["com"]["com_torso"]["ax0"]),
        "torso_y": lp(paths["com"]["com_torso"]["ax1"]),
    }


# -------------------------------------------------------------
# 3. Combine and filter onset sequences
# -------------------------------------------------------------
def compute_combinations(data, fps, thres):
    """Compute all combinations of hand/foot onsets."""
    f = filter_dir_onsets_by_threshold
    k = "both"

    torso_y = data["torso_y"]
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
        "bothhand_y_bothfoot_y_torso_y": f(bothhand_y + bothfoot_y + torso_y, threshold_s=thres, fps=fps),
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
        tempo_info = dance_tempo_estimation_single(tempogram_ab, tempogram_raw, fps, novelty_length, window_size, hop_size, tempi_range)

        bpm_arr = tempo_info["bpm_arr"].flatten()
        tempo_data[seg_key] = {
            **tempo_info,
            "bpm_avg": np.round(np.average(bpm_arr), 2),
            "bpm_mode": stats.mode(bpm_arr)[0],
            "bpm_median": np.median(bpm_arr)
        }

    return tempo_data


# -------------------------------------------------------------
# 6. Main processing loop
# -------------------------------------------------------------
def process_all_files(aist_filelist, anchor_type, mode, fps, window_size, hop_size, tempi_range, save_dir):
    """Master loop to process all files and save per-segment tempo results."""
    result = {}
    count = 0

    for filename in tqdm(aist_filelist):
        paths = get_all_anchor_paths(anchor_type, mode, filename)
        if not os.path.exists(paths["markers"]["left_wrist"]["ax0"]):
            continue

        meta = parse_filename(filename)
        data = load_marker_data(paths)
        combined = compute_combinations(data, fps, thres=0.2)
        resultants = load_resultant(paths, thres=0.2, fps=fps)

        segment_ax = {**combined, **resultants}
        novelty_length = data["left_hand_x"].shape[0]
        
        tempo_data = compute_tempo_for_segments(segment_ax, fps, window_size, hop_size, tempi_range, novelty_length)

        for seg_key, info in tempo_data.items():
            if seg_key not in result:
                result[seg_key] = {k: [] for k in [
                    "filename", "dance_genre", "situation", "camera_id", "dancer_id",
                    "music_id", "choreo_id", "music_tempo", "estimated_bpm_per_window",
                    "magnitude_per_window", "bpm_avg", "bpm_mode", "bpm_median"
                ]}

            result[seg_key]["filename"].append(filename.replace(".pkl", ""))
            for k, v in meta.items():
                result[seg_key][k].append(v)
            result[seg_key]["music_tempo"].append(aist_tempo[meta["music_id"]])
            result[seg_key]["estimated_bpm_per_window"].append(info["bpm_arr"])
            result[seg_key]["magnitude_per_window"].append(info["mag_arr"])
            result[seg_key]["bpm_avg"].append(info["bpm_avg"])
            result[seg_key]["bpm_mode"].append(info["bpm_mode"])
            result[seg_key]["bpm_median"].append(info["bpm_median"])

        count += 1

    print("Total processed:", count)
    # Save results
    for seg_key, df_data in result.items():
        df_seg = pd.DataFrame(df_data)
        fname = f"{anchor_type}/{seg_key}_{mode}.pkl"
        df_seg.to_pickle(os.path.join(save_dir, fname))
       

# -------------------------------------------------------------
# 7. Tempo computation multi segment
# -------------------------------------------------------------   
       
def compute_tempo_for_multi_segments(multi_segment, fps, window_size, hop_size, tempi_range, novelty_length):

    tempo_data = {}
    for seg_key, seg_info in multi_segment.items():

        anchors = []
        tempogram_ab_list = []
        tempogram_raw_list = []
        
        segments = seg_info["segments"]
        segment_names = seg_info["names"]

        for anchor in segments:
            anchors.append(anchor)  # already binary/peak as you like
            temp_ab, temp_raw, _, _ = compute_tempogram(anchor, fps, window_size, hop_size, tempi_range)
            tempogram_ab_list.append(temp_ab)
            tempogram_raw_list.append(temp_raw)

        
        # Use your provided function (returns list of dicts, one per anchor)
        per_anchor = dance_tempo_estimation(
            tempogram_ab_list, tempogram_raw_list,
            fps, novelty_length, window_size, hop_size, tempi_range
        )


        # Build test frequencies (Hz) from anchor-wise median tempos
        test_frequencies = [np.round(d["median_tempo"] / 60.0, 2) for d in per_anchor]
        results = {}
        best_global = {"freq": None, "corr": 0.0, "lag": None}


        for f in test_frequencies:
            for x in anchors:
                best_corr, best_lag, corr, lags = best_alignment(x, f, fps)
                results[f] = (best_corr, best_lag)
                if abs(best_corr) > abs(best_global["corr"]):
                    best_global.update({"freq": f, "corr": best_corr, "lag": best_lag})

        
        # After best_global determined
        if best_global["freq"] is not None:
            best_index = test_frequencies.index(best_global["freq"])
            best_segment_name = segment_names[best_index]
            best_anchor_seq = anchors[best_index]
        else:
            best_segment_name = None
            best_anchor_seq = None
        
        
        # Global tempo in BPM (falls back to anchor median if something odd happens)
        if best_global["freq"] is None:
            # Fallback: use the median of medians (close to V1’s spirit)
            gtempo = float(np.round(np.median([d["median_tempo"] for d in per_anchor]), 2))
        else:
            
            gtempo = float(np.round(best_global["freq"] * 60.0, 2))

        tempo_data[seg_key] = {
            "gtempo": gtempo,
            'best_segment': best_segment_name,
            'anchor_seq': best_anchor_seq,
            "per_anchor": per_anchor,    # contains "median_tempo", "magnitude", "phase", "complex"
            "alignment": {
                "best": best_global,
                "all": results,
            },
        }

    return tempo_data
 
        
# -------------------------------------------------------------
# 8. Main processing loop multi segment
# -------------------------------------------------------------        

def process_all_files_multi_segment(aist_filelist, anchor_type, mode, fps, window_size, hop_size, tempi_range, save_dir):
    """Master loop to process all files and save per-segment tempo results."""
    result = {}
    count = 0

    for filename in tqdm(aist_filelist):
        paths = get_all_anchor_paths(anchor_type, mode, filename)
        if not os.path.exists(paths["markers"]["left_wrist"]["ax0"]):
            continue

        meta = parse_filename(filename)
        data = load_marker_data(paths)
        combined = compute_combinations(data, fps, thres=0.2)
        resultants = load_resultant(paths, thres=0.2, fps=fps)

        segment_ax = {**combined, **resultants}
        multi_segment = build_multi_segment(segment_ax)


        novelty_length = data["left_hand_x"].shape[0]

        tempo_data = compute_tempo_for_multi_segments(multi_segment, fps, window_size, hop_size, tempi_range, novelty_length)

        for seg_key, info in tempo_data.items():
                if seg_key not in result:
                    result[seg_key] = {k: [] for k in [
                        "filename", "dance_genre", "situation", "camera_id", "dancer_id",
                        "music_id", "choreo_id", "music_tempo", "gtempo", 'best_segment_name',
                        "best_anchor_seq"
                    ]}

                result[seg_key]["filename"].append(filename.replace(".pkl", ""))
                for k, v in meta.items():
                    result[seg_key][k].append(v)
                result[seg_key]["music_tempo"].append(aist_tempo[meta["music_id"]])
                result[seg_key]["gtempo"].append(info["gtempo"])
                result[seg_key]["best_segment_name"].append(info["best_segment"])
                result[seg_key]["best_anchor_seq"].append(info["anchor_seq"])


        count += 1

    print("Total processed:", count)
    # Save results
    for seg_key, df_data in result.items():
        df_seg = pd.DataFrame(df_data)
        sub_dir = f"multi/{anchor_type}/"
        os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)
        fname= f"{seg_key}_{mode}.pkl"
        df_seg.to_pickle(os.path.join(save_dir, sub_dir, fname))
        

def build_multi_segment(segment_ax):
    """Build all multi-segment combinations."""
    return {
        'bothhand_y_bothfoot_y_torso_y': {
            "segments": [segment_ax["both_hand_y"], segment_ax["both_foot_y"], segment_ax["torso_y"]],
            "names": ["both_hand_y", "both_foot_y", "torso_y"]
        },
        'bothhand_y_bothfoot_y': {
            "segments": [segment_ax["both_hand_y"], segment_ax["both_foot_y"]],
            "names": ["both_hand_y", "both_foot_y"]
        },
        
        'left_hand_x_right_hand_x': {
            "segments": [segment_ax["left_hand_x"], segment_ax["right_hand_x"]],
            "names": ["left_hand_x", "right_hand_x"]
        },
        
        'left_hand_y_right_hand_y': {
            "segments": [segment_ax["left_hand_y"], segment_ax["right_hand_y"]],
            "names": ["left_hand_y", "right_hand_y"]
        },
        
        'left_foot_x_right_foot_x': {
            "segments": [segment_ax["left_foot_x"], segment_ax["right_foot_x"]],
            "names": ["left_foot_x", "right_foot_x"]
        },
        
        'left_foot_y_right_foot_y': {
            "segments": [segment_ax["left_foot_y"], segment_ax["right_foot_y"]],
            "names": ["left_foot_y", "right_foot_y"]
        },
        
        'left_hand_res_right_hand_res': {
            "segments": [segment_ax["left_hand_resultant"], segment_ax["right_hand_resultant"]],
            "names": ["left_hand_res", "right_hand_res"]
        },
        
        'left_foot_res_right_foot_res': {
            "segments": [segment_ax["left_foot_resultant"], segment_ax["right_foot_resultant"]],
            "names": ["left_foot_res", "right_foot_res"]
        },
        
        # 'leftfoot_xy_rightfoot_xy': {
        #     "segments": [segment_ax["leftfoot_xy"], segment_ax["rightfoot_xy"]],
        #     "names": ["leftfoot_xy", "rightfoot_xy"]
        # },
        # 'left_foot_res_right_foot_res': {
        #     "segments": [segment_ax["left_foot_resultant"], segment_ax["right_foot_resultant"]],
        #     "names": ["left_foot_resultant", "right_foot_resultant"]
        # },
        # 'lefthand_xy_righthand_xy': {
        #     "segments": [segment_ax["lefthand_xy"], segment_ax["righthand_xy"]],
        #     "names": ["lefthand_xy", "righthand_xy"]
        # },
        # 'left_hand_res_right_hand_res': {
        #     "segments": [segment_ax["left_hand_resultant"], segment_ax["right_hand_resultant"]],
        #     "names": ["left_hand_resultant", "right_hand_resultant"]
        # },
        # 'bothfoot_x_bothfoot_y': {
        #     "segments": [segment_ax["both_foot_x"], segment_ax["both_foot_y"]],
        #     "names": ["both_foot_x", "both_foot_y"]
        # },
        # 'bothhand_x_bothfoot_x': {
        #     "segments": [segment_ax["both_hand_x"], segment_ax["both_foot_x"]],
        #     "names": ["both_hand_x", "both_foot_x"]
        # },
        # 'bothhand_x_bothhand_y': {
        #     "segments": [segment_ax["both_hand_x"], segment_ax["both_hand_y"]],
        #     "names": ["both_hand_x", "both_hand_y"]
        # },
        # 'both_hand_res_both_foot_res': {
        #     "segments": [segment_ax["both_hand_resultant"], segment_ax["both_foot_resultant"]],
        #     "names": ["both_hand_resultant", "both_foot_resultant"]
        # }
    }
