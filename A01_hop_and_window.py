import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

# from utils import compute_tempo as ctempo
from utils.anchor_io import *
from utils.dance_tempo_pipeline import *



anchor_type1 = "anchor_zero"
anchor_type2 = "anchor_peak"
anchor_type3 = "anchor_energy"
mode = "uni"
fps = 60    # ---------------------------------------------------------------------------------
# w_sec = 5
# h_sec = w_sec/2
# window_size = int(fps*w_sec)
# hop_size = int(fps*h_sec)

a = 45 
b = 140
tempi_range = np.arange(a,b,1)

# output_dir = f"tempo_estimation_output"
# create_output_dir(output_dir, f"tempo_{a}_{b}")
# save_dir = os.path.join(output_dir, f"tempo_{a}_{b}/")


aist2d_path = "./aist_dataset/aist_annotation/keypoints2d"
aist_filelist = os.listdir(aist2d_path)


hop_list = [0.25, 0.5, 0.75, 1.0]   # percent
window_list = [1, 2, 3, 4, 5, 6, 7, 8]  # seconds

for hop in hop_list:
    for window in window_list:
        window_size = int(fps*window)
        hop_size = int(window_size * hop) 
        
        output_dir = f"tempo_estimation_output_H{hop}_W{window}"
        create_output_dir(output_dir, f"tempo_{a}_{b}")
        save_dir = os.path.join(output_dir, f"tempo_{a}_{b}/")
        
        
        print(f"Processing with window size: {window} and hop size: {hop}")
        process_all_files_multi_segment(aist_filelist, anchor_type1, mode, fps, window_size, hop_size, tempi_range, save_dir)