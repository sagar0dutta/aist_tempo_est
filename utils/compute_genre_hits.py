import os
import pickle
import json
import numpy as np
import pandas as pd


# Load genre mappings
with open("genreID_count_mapping.json", "r") as file:
    genre_Tcount = json.load(file)

with open("genre_symbols_mapping.json", "r") as file:
    genre_name = json.load(file)
    
    
def save_pickle(filepath, data):
    # filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
        
def compute_genre_hits(df_seg, mode, anchor_type, method, output_path):

    read_dir = os.path.join(output_path, "eval_data", method)

    # fname = f"{anchor_type}_{mode}.pkl"
    # fpath = os.path.join(read_dir, fname)

    # df_eval = pd.read_pickle(fpath)

    fpath1 = os.path.join(output_path, anchor_type, "left_hand_y_uni.pkl")    # For reference only
    df_ref = pd.read_pickle(fpath1)

    # Initialize base dataframe with genre names and total counts
    final_df = pd.DataFrame(list(genre_name.items()), columns=["dance_genre", "genre"])
    final_df["total"] = final_df["dance_genre"].map(genre_Tcount)

    # Convert hit indices to list (in case NumPy array)
    hit_indices = df_seg["hit_idx"].tolist()

    # Extract DataFrame of hits
    hit_df = df_ref.iloc[hit_indices]

    # Count hits per genre
    genre_counts = (
        hit_df.groupby("dance_genre")
        .size()
        .reset_index(name="hit_count")
    )

    # Merge with base dataframe
    hits_genrewise_df = final_df.merge(genre_counts, on="dance_genre", how="left")
    hits_genrewise_df["hit_count"] = hits_genrewise_df["hit_count"].fillna(0).astype(int)

    # Compute percentages
    hits_genrewise_df["hit_percentage"] = (
        (hits_genrewise_df["hit_count"] / hits_genrewise_df["total"]) * 100
    ).round(2)

    # Select and order final columns
    hits_genrewise_df = hits_genrewise_df[["genre", "total", "hit_count", "hit_percentage"]]

    # Optionally save
    sv_path = os.path.join(read_dir, f"hits_genrewise_{anchor_type}_{mode}.pkl")
    save_pickle(sv_path, hits_genrewise_df)

    return hits_genrewise_df