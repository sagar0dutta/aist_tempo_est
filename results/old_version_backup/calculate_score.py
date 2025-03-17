import os
import numpy as np
import seaborn as sns
import pandas as pd
from dance_evaluation import *
from collections import defaultdict


def calc_score(mode, read_path, save_path, tol_type = "abs", tolerance = 8):


    # root_dir = f"aist_pos1s/tempo_{a}_{b}"
    df_results = pd.read_pickle(read_path)

    ref = df_results["music_tempo"].to_numpy()
    bpm_avg_x = df_results[f"bpm_avg_x"].to_numpy()
    bpm_avg_y = df_results[f"bpm_avg_y"].to_numpy()
    bpm_avg_xy = df_results[f"bpm_avg_xy"].to_numpy()

    bpm_mode_x = df_results[f"bpm_mode_x"].to_numpy()
    bpm_mode_y = df_results[f"bpm_mode_y"].to_numpy()
    bpm_mode_xy = df_results[f"bpm_mode_xy"].to_numpy()

    bpm_median_x = df_results[f"bpm_median_x"].to_numpy()
    bpm_median_y = df_results[f"bpm_median_y"].to_numpy()
    bpm_median_xy = df_results[f"bpm_median_xy"].to_numpy()

    # Data for each experiment
    experiments = {
    "1s": [bpm_avg_x, bpm_avg_y, bpm_avg_xy, 
           bpm_mode_x, bpm_mode_y, bpm_mode_xy,
           bpm_median_x, bpm_median_y, bpm_median_xy],
            }
    axes = ["bpm_avg_x", "bpm_avg_y", "bpm_avg_xy",
            "bpm_mode_x", "bpm_mode_y", "bpm_mode_xy",
            "bpm_median_x", "bpm_median_y", "bpm_median_xy"]

    # results = {"direction": [], "axis": [], "acc1": [], "acc2": [], "acc3": [],
    #         "hits_idx": [], "hits_dbl_idx": [], "hits_hf_idx": []}
    
    results = defaultdict(list)

    
    for exp_name, data in experiments.items():
        for axis_name, calculated in zip(axes, data):
            metrics = calculate_metrics_with_oe(ref, calculated, tol_type = tol_type, tolerance = tolerance)
            
            results["mode"].append(mode)
            results["axis"].append(axis_name)
            results["acc1"].append(metrics["acc1"])
            results["acc2"].append(metrics["acc2"]) # double/ half
            results["acc3"].append(metrics["acc3"]) # 1x 2x 3x 0.5x 0.33x
            results["base_tempo"].append(metrics["base_tempo"])
            
            results["hits_idx"].append(metrics["hits_idx"].tolist())
            results["hits_dbl_idx"].append(metrics["hits_dbl_idx"].tolist())
            results["hits_hf_idx"].append(metrics["hits_hf_idx"].tolist())
            
            results["bpm_hits_idx"].append(metrics["bpm_hits_idx"].tolist())
            results["bpm_hits_dbl_idx"].append(metrics["bpm_hits_dbl_idx"].tolist())
            results["bpm_hits_hf_idx"].append(metrics["bpm_hits_hf_idx"].tolist())

    results_df = pd.DataFrame(results)
    results_df.to_pickle(save_path)
    print("Score saved to", save_path)
    
    
def updated_calc_score(mode, read_path, save_path, tol_type = "abs", tolerance = 8):

    df_results = pd.read_pickle(read_path)

    ref = df_results["music_tempo"].to_numpy()
    bpm_avg = df_results[f"bpm_avg"].to_numpy()
    bpm_mode = df_results[f"bpm_mode"].to_numpy()
    bpm_median = df_results[f"bpm_median"].to_numpy()


    # Data for each experiment
    experiments = {
    "1s": [bpm_avg,  
           bpm_mode, 
           bpm_median, ],
            }
    axes = ["bpm_avg", 
            "bpm_mode", 
            "bpm_median",]

    
    results = defaultdict(list)

    for exp_name, data in experiments.items():
        for axis_name, calculated in zip(axes, data):
            metrics = calculate_metrics_with_oe(ref, calculated, tol_type = tol_type, tolerance = tolerance)
            
            results["mode"].append(mode)
            results["axis"].append(axis_name)
            results["acc1"].append(metrics["acc1"])
            results["acc2"].append(metrics["acc2"]) # double/ half
            results["acc3"].append(metrics["acc3"]) # 1x 2x 3x 0.5x 0.33x
            
            results["base_tempo"].append(metrics["base_tempo"])
            results["hits_idx"].append(metrics["hits_idx"].tolist())
            results["hits_dbl_idx"].append(metrics["hits_dbl_idx"].tolist())
            results["hits_hf_idx"].append(metrics["hits_hf_idx"].tolist())
            
            results["bpm_hits_idx"].append(metrics["bpm_hits_idx"].tolist())
            results["bpm_hits_dbl_idx"].append(metrics["bpm_hits_dbl_idx"].tolist())
            results["bpm_hits_hf_idx"].append(metrics["bpm_hits_hf_idx"].tolist())

    results_df = pd.DataFrame(results)
    results_df.to_pickle(save_path)
    print("Score saved to", save_path)