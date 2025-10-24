

import os
import pickle 
import pandas as pd
import numpy as np






def compute_dts(
    ref_bpm,
    estimated_bpm,
    tau=0.13,
    mode="one"
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
    mode : {"one", "many"} 
        “one”: treat `estimated_bpm` as a flat sequence.
        “many”: pick, for each i, the candidate closest to ref_bpm[i]. For best of two

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
    if mode == "many":
        chosen = np.array([
            min(cands, key=lambda b: min(
            abs(b - ref_bpm[i]),
            abs(b - 0.5 * ref_bpm[i]),
            abs(b - 2.0 * ref_bpm[i])
        ))
        for i, cands in enumerate(estimated_bpm)
        ], dtype=float)
    
    elif mode == "one":
        chosen = np.asarray(estimated_bpm, dtype=float)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'one' or 'many'.")

    # DTS core ------------------------------------------------------
    e = np.log2(chosen / ref_bpm)
    # distance from nearest of -1, 0, +1
    d = np.abs(e[:, None] - np.array([-1.0, 0.0, 1.0])).min(axis=1)
    # clip by tolerance and convert to score
    d_clip = np.minimum(d, tau)
    dts    = 1.0 - d_clip / tau

    accuracy = (dts > 0.0).mean() * 100
    
    # hits ----------------------------------------------------------
    hit_mask = dts > 0.0          # inside ±tau band
    hit_idx = np.nonzero(hit_mask)[0]
    ref_hit_bpm = ref_bpm[hit_idx]
    
    return accuracy, hit_idx, ref_hit_bpm

##-------------------------------------------------
## Evaluation mmethod best of n
##-------------------------------------------------

def compute_dts_bon(
    ref_bpm,
    estimated_bpm,
    tau=0.13,
    mode="one"
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
    mode : {"one", "many"} 
        “one”: treat `estimated_bpm` as a flat sequence.
        “many”: pick, for each i, the candidate closest to ref_bpm[i]. For best of two

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

    if mode == "many":
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

    elif mode == "one":
        chosen = [(float(b), None) for b in np.asarray(estimated_bpm, dtype=float)]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'one' or 'many'.")

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
    ref_hit_bpm = ref_bpm[hit_idx]
    
    return accuracy, hit_idx, ref_hit_bpm, chosen




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
    
    score_data = {}
    json_data = {}
    output_path = f"./tempo_estimation_output/tempo_{a}_{b}/"

    score_data["bpm_median"] = {}
    json_data["bpm_median"] = {}
    
    for idx, f_name in enumerate(segment_keys):
        
        f_path = output_path + f"{anchor_type}/{f_name}_{mode}.pkl"        # _W{w_sec}_H{h_sec}_{a}_{b}
        df_ax = pd.read_pickle(f_path)

        ref = df_ax["music_tempo"].to_numpy()
        dts_acc, hit_idx, ref_hit_bpm = compute_dts(ref, np.asarray(df_ax["bpm_median"]), tau=tolerance, mode = "one")
        
        json_data["bpm_median"][f_name] = {"accuracy": np.round(dts_acc, 2),
                                            "hit_index": hit_idx,
                                            "ref_hit_bpm": ref_hit_bpm}
                            

    #### Save the score data to a pickle file
    save_dir = os.path.join(output_path, "eval_data", "single")
    os.makedirs(save_dir, exist_ok=True)
    
    fname = f"{anchor_type}_{mode}.pkl"
    fpath = os.path.join(save_dir, fname)
    save_to_pickle(fpath, score_data)
    
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
        ref  = data["music_tempo"].to_numpy()
        accuracy, hit_idx, ref_hit_bpm = compute_dts(ref, data["gtempo"].to_numpy(),
                                            tau=tolerance, mode="one")
        
        acc[seg] = round(accuracy, 2)
        json_data[seg] = {"acc": accuracy, "hit_idx": hit_idx, "ref_hit_bpm": ref_hit_bpm}
        

    #### Save the score data to a pickle file
    output_path = f"./tempo_estimation_output/tempo_{a}_{b}/"
    save_dir = os.path.join(output_path, "eval_data", "multi")
    os.makedirs(save_dir, exist_ok=True)
    
    fname = f"{anchor_type}_{mode}.pkl"
    fpath = os.path.join(save_dir, fname)
    save_to_pickle(fpath, json_data)
    
    return json_data



###----------------------------------------------------------------------
## Evaluation for best of n
###----------------------------------------------------------------------

def eval_best_of_n(segment_names, a, b, mode, anchor_type, tolerance=0.13):
    """
    Evaluate 'best-of-n' accuracy across multiple segments.

    Args:
        segment_names (list[str]): List of segment names (e.g., ["both_hand_y", "both_foot_y", "torso_y"])
        a, b (int/float): Tempo range parameters
        mode (str): Mode name (e.g., "pos", "vel")
        anchor_type (str): Anchor type (e.g., "anchor_zero", "anchor_peak")
        tolerance (float): Allowed tempo deviation (default 0.13)
    """
    output_path = f"./tempo_estimation_output/tempo_{a}_{b}/"
    score_data = {}
    bpm_candidates = []

    # Load all DataFrames dynamically for each segment
    dfs = []
    for seg in segment_names:
        fpath = os.path.join(output_path, anchor_type, f"{seg}_{mode}.pkl")
        dfs.append(pd.read_pickle(fpath))

    # Assume all DataFrames are aligned (same rows for each recording)
    num_rows = dfs[0].shape[0]
    ref = dfs[0]["music_tempo"].to_numpy()  # Reference tempo (same across all)

    # Build candidate BPM tuples across n segments
    for i in range(num_rows):
        bpm_tuple = tuple(df.iloc[i]["bpm_median"] for df in dfs)
        bpm_candidates.append(bpm_tuple)

    # Compute best-of-n accuracy
    acc, hit_idx, ref_hit_bpm, chosen = compute_dts_bon(ref, bpm_candidates, tau=tolerance, mode="many")

    # Save results
    json_data = {
        "accuracy": acc,
        "hit_idx": hit_idx,
        "hit_ref_bpm": ref_hit_bpm,
        "chosen_candidated_bpm": chosen,
    }

    save_dir = os.path.join(output_path, "eval_data", "best_of_n")
    os.makedirs(save_dir, exist_ok=True)
    
    fname = f"{anchor_type}_{mode}.pkl"
    fpath = os.path.join(save_dir, fname)
    save_to_pickle(fpath, json_data)

    return json_data





### Utility functions for pickle handling
def load_pickle(filepath):
    with open(filepath, "rb") as f:
        json_data = pickle.load(f)
    return json_data

def save_to_pickle(filepath, data):
    # filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)