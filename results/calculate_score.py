import os
import numpy as np
import seaborn as sns
import pandas as pd
from dance_evaluation import *
import matplotlib.pyplot as plt



def calc_score(mode, read_path, save_path):


    # root_dir = f"aist_pos1s/tempo_{a}_{b}"
    df_results = pd.read_csv(read_path)

    ref = df_results["music_tempo"].to_numpy()
    tempo_x_z1 = df_results[f"X_a"].to_numpy()
    tempo_y_z1 = df_results[f"Y_a"].to_numpy()
    tempo_mode_z1 = df_results[f"bpm_mode"].to_numpy()
    tempo_median_z1 = df_results[f"bpm_median"].to_numpy()

    # Data for each experiment
    experiments = {
        "1s": [tempo_x_z1, tempo_y_z1, tempo_mode_z1, tempo_median_z1],
        
    }
    axes = ["X", "Y", "mode", "median"]

    results = {"direction": [], "axis": [], "acc1": [], "acc2": [], "acc3": [],
            "hits_idx": [], "hits_dbl_idx": [], "hits_hf_idx": []}

    tolerance = 8
    for exp_name, data in experiments.items():
        for axis_name, calculated in zip(axes, data):
            metrics, hits_idx, hits_dbl_idx, hits_hf_idx = calculate_metrics_with_oe(ref, calculated, tolerance = tolerance)
            
            results["direction"].append(mode)
            results["axis"].append(axis_name)
            results["acc1"].append(metrics["acc1"])
            results["acc2"].append(metrics["acc2"]) # double/ half
            results["acc3"].append(metrics["acc3"]) # 1x 2x 3x 0.5x 0.33x
            
            results["hits_idx"].append(hits_idx.tolist())
            results["hits_dbl_idx"].append(hits_dbl_idx.tolist())
            results["hits_hf_idx"].append(hits_hf_idx.tolist())

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path)
    print("Score saved to", save_path)