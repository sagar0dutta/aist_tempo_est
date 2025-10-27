

import os
import json
import pickle 
import numpy as np
import pandas as pd
from collections import defaultdict





def compute_dts(
    ref_bpm,
    estimated_bpm,
    tau=0.13,
    ):
    """
    Continuous Dance-Tempo Score (DTS), with support for
    either single estimates (mode="one") or multiple
    candidates per frame (mode="many").

    Parameters
    ----------
    ref_bpm : array-like, shape (n,)
        Ground-truth musical tempo in BPM.
    estimated_bpm : 
        If mode="one": array-like, shape (n,)
        If mode="many": iterable of length-n, each element
                        is an iterable of candidate BPMs.
    tau : float, optional
        Tolerance in octaves (0.06 ≈ 4 %).


    Returns
    -------
    dts : ndarray, shape (n,)
        Scores in [0, 1] (1 = perfect, 0 = miss ≥ τ octaves away).
    e : ndarray, shape (n,)
        Raw octave errors log₂(estimate/ref).
    d : ndarray, shape (n,)
        Wrapped distance to {-1, 0, +1} before clipping.
    """
    ref_bpm = np.asarray(ref_bpm, dtype=float)

    # select a single estimate per index if needed
    # if mode == "many":
    #     chosen = np.array([
    #         min(cands, key=lambda b: min(
    #         abs(b - ref_bpm[i]),
    #         abs(b - 0.5 * ref_bpm[i]),
    #         abs(b - 2.0 * ref_bpm[i])
    #     ))
    #     for i, cands in enumerate(estimated_bpm)
    #     ], dtype=float)
    
    # elif mode == "one":
    #     chosen = np.asarray(estimated_bpm, dtype=float)
    # else:
    #     raise ValueError(f"Unknown mode: {mode!r}. Use 'one' or 'many'.")

    # DTS core ------------------------------------------------------
    e = np.log2(estimated_bpm / ref_bpm)
    # distance from nearest of -1, 0, +1
    d = np.abs(e[:, None] - np.array([-1.0, 0.0, 1.0])).min(axis=1)
    # clip by tolerance and convert to score
    d_clip = np.minimum(d, tau)
    dts    = 1.0 - d_clip / tau

    accuracy = (dts > 0.0).mean() * 100
    
    # hits ----------------------------------------------------------
    hit_mask = dts > 0.0          # inside ±tau band
    hit_idx = np.nonzero(hit_mask)[0]
    hit_ref_bpm = ref_bpm[hit_idx]
    
    json_data = {
        "accuracy": accuracy,
        "hit_idx": hit_idx,
        "hit_ref_bpm": hit_ref_bpm,
        "dts": dts
    }
    
    
    return json_data

##-------------------------------------------------
## Evaluation mmethod best of n
##-------------------------------------------------

def compute_dts_bon(
    ref_bpm,
    estimated_bpm,
    tau=0.13,
    ):
    """
    Continuous Dance-Tempo Score (DTS), with support for
    either single estimates (mode="one") or multiple
    candidates per frame (mode="many").

    Parameters
    ----------
    ref_bpm : array-like, shape (n,)
        Ground-truth musical tempo in BPM.
    estimated_bpm : 
        If mode="one": array-like, shape (n,)
        If mode="many": iterable of length-n, each element
                        is an iterable of candidate BPMs.
    tau : float, optional
        Tolerance in octaves (0.06 ≈ 4 %).
    

    Returns
    -------
    dts : ndarray, shape (n,)
        Scores in [0, 1] (1 = perfect, 0 = miss ≥ τ octaves away).
    e : ndarray, shape (n,)
        Raw octave errors log₂(estimate/ref).
    d : ndarray, shape (n,)
        Wrapped distance to {-1, 0, +1} before clipping.
    """
    ref_bpm = np.asarray(ref_bpm, dtype=float)

    body_parts = ["hand", "foot", "torso"]

    # if mode == "many":
    chosen = []
    for i, cands in enumerate(estimated_bpm):  # e.g. (bpm_hand, bpm_foot, bpm_torso)
        ref = ref_bpm[i]
        diffs = [
            min(abs(b - ref), abs(b - 0.5 * ref), abs(b - 2.0 * ref))
            for b in cands
        ]
        idx_min = int(np.argmin(diffs))  # index of best match
        chosen_bpm = cands[idx_min]
        chosen_part = body_parts[idx_min]
        chosen.append((chosen_bpm, chosen_part))

    # elif mode == "one":
    #     chosen = [(float(b), None) for b in np.asarray(estimated_bpm, dtype=float)]
    # else:
    #     raise ValueError(f"Unknown mode: {mode!r}. Use 'one' or 'many'.")

    chosen_bpm = np.array([c[0] for c in chosen], dtype=float)
    # DTS core ------------------------------------------------------
    e = np.log2(chosen_bpm / ref_bpm)
    # distance from nearest of -1, 0, +1
    d = np.abs(e[:, None] - np.array([-1.0, 0.0, 1.0])).min(axis=1)
    # clip by tolerance and convert to score
    d_clip = np.minimum(d, tau)
    dts    = 1.0 - d_clip / tau

    accuracy = (dts > 0.0).mean() * 100
    
    # hits ----------------------------------------------------------
    hit_mask = dts > 0.0          # inside ±tau band
    hit_idx = np.nonzero(hit_mask)[0]
    hit_ref_bpm = ref_bpm[hit_idx]
    
    json_data = {
        "accuracy": accuracy,
        "hit_idx": hit_idx,
        "hit_ref_bpm": hit_ref_bpm,
        "hit_selctd_bpm": chosen,
        "dts": dts
    }
    
    return json_data




##----------------------------------------------------------------------
## Evaluation for single anchor sequence
##----------------------------------------------------------------------

def evaluation_single(a, b, mode, anchor_type, tolerance=0.13):

    segment_keys = [
                    "torso_x","torso_y",
                    "left_hand_x", "right_hand_x", "left_hand_y", "right_hand_y",   # singular
                    "left_foot_x", "right_foot_x", "left_foot_y", "right_foot_y",   # singular
                    
                    "lefthand_xy", "righthand_xy", "leftfoot_xy", "rightfoot_xy",   # singular | 35, 40, 34, 36
                    "left_hand_resultant", "right_hand_resultant", "left_foot_resultant", "right_foot_resultant",   # singular | 18,20,17,17 %
                    
                    "both_hand_x", "both_hand_y", "both_foot_x", "both_foot_y",
                     "both_hand_resultant", "both_foot_resultant", # resultant of x and y onsets
                    
                    "bothhand_x_bothfoot_x", "bothhand_y_bothfoot_y",
                    "lefthand_xy_righthand_xy", "leftfoot_xy_rightfoot_xy",
                    "bothhand_x_bothhand_y", "bothfoot_x_bothfoot_y", 
                    "bothhand_y_bothfoot_y_torso_y",
                    ] 
    

    json_data = {}
    json_data["bpm_median"] = {}
    output_path = f"./tempo_estimation_output/tempo_{a}_{b}/"

    
    for idx, f_name in enumerate(segment_keys):
        
        f_path = output_path + f"{anchor_type}/{f_name}_{mode}.pkl"        # _W{w_sec}_H{h_sec}_{a}_{b}
        df_ax = pd.read_pickle(f_path)

        ref = df_ax["music_tempo"].to_numpy()
        dts_data = compute_dts(ref, np.asarray(df_ax["bpm_median"]), tau=tolerance)
        
        accuracy     = dts_data["accuracy"]
        dts          = dts_data["dts"]
        hit_idx      = dts_data["hit_idx"]
        hit_ref_bpm  = dts_data["hit_ref_bpm"]
        
        
        json_data["bpm_median"][f_name] = {"accuracy": np.round(accuracy, 2),
                                            "dts": dts,
                                            "hit_index": hit_idx,
                                            "ref_hit_bpm": hit_ref_bpm,
                                            }
                            

    #### Save the score data to a pickle file
    save_dir = os.path.join(output_path, "eval_data", "single")
    os.makedirs(save_dir, exist_ok=True)
    
    fname = f"{anchor_type}_{mode}.pkl"
    fpath = os.path.join(save_dir, fname)
    save_to_pickle(fpath, json_data)
    
    return json_data

###----------------------------------------------------------------------
## Evaluation for multi segments
###----------------------------------------------------------------------
def evaluation_multi_segment(anchor_type, mode, a, b, tolerance=0.13):
    acc = {}
    json_data = {}
    root_dir = "./tempo_estimation_output"
    anchor_dir = os.path.join(root_dir, f"tempo_{a}_{b}", "multi", anchor_type)
    
    multi_segment = [
            "bothhand_y_bothfoot_y",
            "leftfoot_xy_rightfoot_xy",
            "left_foot_res_right_foot_res",
            "lefthand_xy_righthand_xy",
            "left_hand_res_right_hand_res",
            "bothfoot_x_bothfoot_y",
            "bothhand_x_bothfoot_x",
            "bothhand_x_bothhand_y",
            "both_hand_res_both_foot_res",
            "bothhand_y_bothfoot_y_torso_y",
            ]
    
    for seg in multi_segment:

        file_name = f"{seg}_{mode}.pkl"
        file_path = os.path.join(anchor_dir, file_name)
        
        data = load_pickle(file_path)
        ref_bpm  = data["music_tempo"].to_numpy()
        best_seg_names = data["best_segment_name"].to_numpy()
        dance_genre = data["dance_genre"].to_numpy()
        fnames = data["filename"].to_numpy()
        
        
        dts_data = compute_dts(ref_bpm, data["gtempo"].to_numpy(), tau=tolerance)
        
        accuracy     = dts_data["accuracy"]
        dts          = dts_data["dts"]
        hit_idx      = dts_data["hit_idx"]
        hit_ref_bpm  = dts_data["hit_ref_bpm"]
        
        
        hit_seg_names = best_seg_names[hit_idx]
        hit_dance_genre = dance_genre[hit_idx]
        hit_fname = fnames[hit_idx]
     
        
        acc[seg] = round(accuracy, 2)
        json_data[seg] = {"acc": accuracy, "dts": dts ,"hit_idx": hit_idx, 
                          "hit_ref_bpm": hit_ref_bpm, "hit_seg_names": hit_seg_names,
                          "hit_genres":hit_dance_genre, 'filename': hit_fname}
        

    #### Save the score data to a pickle file
    output_path = f"./tempo_estimation_output/tempo_{a}_{b}/"
    save_dir = os.path.join(output_path, "eval_data", "multi")
    os.makedirs(save_dir, exist_ok=True)
    
    fname = f"{anchor_type}_{mode}.pkl"
    fpath = os.path.join(save_dir, fname)
    save_to_pickle(fpath, json_data)
    
    return json_data


####
def eval_best_of_n(segment_groups, a, b, mode, anchor_type, tolerance=0.13):
    """
    Evaluate 'best-of-n' accuracy across multiple segment combinations.

    Args:
        segment_groups (list[list[str]]): List of segment name lists.
            e.g., [["both_hand_y", "both_foot_y"], ["both_hand_y", "both_foot_y", "torso_y"]]
        a, b (int/float): Tempo range parameters.
        mode (str): Mode name (e.g., "pos", "vel").
        anchor_type (str): Anchor type (e.g., "anchor_zero", "anchor_peak").
        tolerance (float): Allowed tempo deviation (default 0.13).
    """
    output_path = f"./tempo_estimation_output/tempo_{a}_{b}/"
    json_data = {}

    for seg_list in segment_groups:
        # Load all DataFrames for the current segment combination
        dfs = []
        for seg in seg_list:
            fpath = os.path.join(output_path, anchor_type, f"{seg}_{mode}.pkl")
            if not os.path.exists(fpath):
                print(f"Warning: File not found — {fpath}")
                continue
            dfs.append(pd.read_pickle(fpath))

        if not dfs:
            print(f"Skipping {seg_list}: no valid data files found.")
            continue

        # Assume all DataFrames are aligned
        num_rows = dfs[0].shape[0]
        ref = dfs[0]["music_tempo"].to_numpy()
        dance_genres = dfs[0]["dance_genre"].to_numpy()

        # Build candidate BPM tuples across n segments
        bpm_candidates = [
            tuple(df.iloc[i]["bpm_median"] for df in dfs)
            for i in range(num_rows)
        ]

        # Compute best-of-n accuracy
        dts_data = compute_dts_bon(ref, bpm_candidates, tau=tolerance)
        
        accuracy     = dts_data["accuracy"]
        dts          = dts_data["dts"]
        hit_idx      = dts_data["hit_idx"]
        hit_ref_bpm  = dts_data["hit_ref_bpm"]
        hit_selctd_bpm  = dts_data["hit_selctd_bpm"]
        

        hit_dance_genres = dance_genres[hit_idx]
        
        # Create merged segment name
        merge_seg_names = "_".join(seg_list)

        # Store results
        json_data[merge_seg_names] = {
            "accuracy": accuracy,
            "dts": dts,
            "hit_idx": hit_idx,
            "hit_ref_bpm": hit_ref_bpm,
            "hit_genres": hit_dance_genres,
            "hit_selected_bpm": hit_selctd_bpm,
        }

    # Save results to pickle
    save_dir = os.path.join(output_path, "eval_data", "best_of_n")
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{anchor_type}_{mode}.pkl"
    fpath = os.path.join(save_dir, fname)
    save_to_pickle(fpath, json_data)

    return json_data



# --- Load genre mapping -------------------------------------------------
with open("genre_symbols_mapping.json", "r") as file:
    genre_id_to_name = json.load(file)

with open("genreID_count_mapping.json", "r") as file:
    genre_Tcount = json.load(file)


def find_body_contribution_best_of_n(chosen, hit_genres_id):
    # --- Map genres only for successful hits -------------------------------
    genre_part_map = defaultdict(list)

    # Iterate only over correct hits
    for hit_idx, genre_id in enumerate(hit_genres_id):
        # The same index in `chosen` corresponds to a correct hit
        bpm, part = chosen[hit_idx]
        genre = genre_id_to_name[str(genre_id)]  # ensure JSON string keys
        genre_part_map[genre].append(part)

    # --- Convert to DataFrame ----------------------------------------------
    genre_part_df = pd.DataFrame([
        {"genre": genre, "body_part": part}
        for genre, parts in genre_part_map.items()
        for part in parts
    ])

    # --- Count frequency per genre -----------------------------------------
    part_counts = (
        genre_part_df.groupby(["genre", "body_part"])
        .size()
        .reset_index(name="count")
        .sort_values(["genre", "count"], ascending=[True, False])
    )


    # --- Normalize by total sequences per genre ----------------------------
    # genre_Tcount_named = {genre_id_to_name [k]: v for k, v in genre_Tcount.items()}
    # part_counts["total_per_genre"] = part_counts["genre"].map(genre_Tcount_named)
    # part_counts["percentage"] = (part_counts["count"] / part_counts["total_per_genre"] * 100).round(2)


    # --- Normalize by total hits per genre (not total sequences) ----------
    genre_totals = part_counts.groupby("genre")["count"].transform("sum")
    part_counts["percentage"] = (part_counts["count"] / genre_totals * 100).round(2)

    return part_counts



def find_body_contribution_multi(hit_genres_multi, hit_segment_names):
    """
    Compute per-genre body-part contribution among correctly estimated tempos.
    
    Args:
        hit_genres_multi (list): List of genre IDs for correct hits.
        hit_segment_names (list): List of segment names (same length as hit_genres_multi).
        genre_id_to_name (dict): Mapping from genre ID (e.g., 'gHO') to readable name ('House').

    Returns:
        pd.DataFrame: Columns = ['genre', 'body_part', 'count', 'percentage']
    """

    # --- Map segments to genres for correct hits --------------------------
    genre_part_map = defaultdict(list)
    for genre_id, seg_name in zip(hit_genres_multi, hit_segment_names):
        genre = genre_id_to_name[str(genre_id)]  # ensure key type consistency
        genre_part_map[genre].append(seg_name)

    # --- Build DataFrame --------------------------------------------------
    genre_part_df = pd.DataFrame([
        {"genre": genre, "body_part": seg}
        for genre, seg_list in genre_part_map.items()
        for seg in seg_list
    ])

    # --- Count frequency per genre ---------------------------------------
    part_counts = (
        genre_part_df.groupby(["genre", "body_part"])
        .size()
        .reset_index(name="count")
        .sort_values(["genre", "count"], ascending=[True, False])
    )
    
    # --- Normalize by total sequences per genre ----------------------------
    # genre_Tcount_named = {genre_id_to_name [k]: v for k, v in genre_Tcount.items()}
    # part_counts["total_per_genre"] = part_counts["genre"].map(genre_Tcount_named)
    # part_counts["percentage"] = (part_counts["count"] / part_counts["total_per_genre"] * 100).round(2)

    # --- Normalize by total hits per genre -------------------------------
    genre_totals = part_counts.groupby("genre")["count"].transform("sum")
    part_counts["percentage"] = (part_counts["count"] / genre_totals * 100).round(2)

    return part_counts






### Utility functions for pickle handling
def load_pickle(filepath):
    with open(filepath, "rb") as f:
        json_data = pickle.load(f)
    return json_data

def save_to_pickle(filepath, data):
    # filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)