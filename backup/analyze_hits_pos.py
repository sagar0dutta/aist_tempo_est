import os
import glob
import ast
import json
import numpy as np
import seaborn as sns
import pandas as pd
from dance_evaluation import *
import matplotlib.pyplot as plt

class HitsAnalysis:
    def __init__(self, a, b, mode, norm_mode, foot1s_name, hand1s_name):
        self.a = a
        self.b = b
        self.total = 0
        self.mode = mode
        self.norm_mode = norm_mode
        self.foot1s_name = foot1s_name
        self.hand1s_name = hand1s_name
        self.side_hand = self.hand1s_name.split("_")[0].capitalize()
        self.side_foot = self.foot1s_name.split("_")[0].capitalize()
        self.foot_1s_df = None
        self.genre_Tcount = None
        self.genre_name = None
    
    def load_files(self):

        self.foot_1S_df = pd.read_csv(f"./aist_pos1s/{self.norm_mode}/tempo_{self.a}_{self.b}/foot/{self.foot1s_name}_zero_{self.mode}_{self.a}_{self.b}.csv")
        foot_2S_df = pd.read_csv(f"./aist_pos2s/{self.norm_mode}/tempo_{self.a}_{self.b}/foot/foot_zero_{self.mode}_{self.a}_{self.b}.csv")

        foot_1s_score = pd.read_csv(f"./aist_pos1s/{self.norm_mode}/tempo_{self.a}_{self.b}/score/foot/{self.foot1s_name}_{self.mode}_{self.a}_{self.b}_score.csv")
        hand_1s_score = pd.read_csv(f"./aist_pos1s/{self.norm_mode}/tempo_{self.a}_{self.b}/score/hand/{self.hand1s_name}_{self.mode}_{self.a}_{self.b}_score.csv")

        foot_2s_score = pd.read_csv(f"./aist_pos2s/{self.norm_mode}/tempo_{self.a}_{self.b}/score/foot/foot_{self.mode}_{self.a}_{self.b}_score.csv")
        hand_2s_score = pd.read_csv(f"./aist_pos2s/{self.norm_mode}/tempo_{self.a}_{self.b}/score/hand/hand_{self.mode}_{self.a}_{self.b}_score.csv")

        foot_1s_score["hits_idx"] = foot_1s_score["hits_idx"].apply(ast.literal_eval)   # convert string to list
        hand_1s_score["hits_idx"] = hand_1s_score["hits_idx"].apply(ast.literal_eval)

        foot_2s_score["hits_idx"] = foot_2s_score["hits_idx"].apply(ast.literal_eval)
        hand_2s_score["hits_idx"] = hand_2s_score["hits_idx"].apply(ast.literal_eval)

        self.total = self.foot_1S_df.shape[0]

        # genre id and total count mapping
        json_filename = "genreID_count_mapping.json"
        with open(json_filename, "r") as file:
            self.genre_Tcount = json.load(file)
            
        # genre id and name mapping
        json_filename = "genre_symbols_mapping.json"
        with open(json_filename, "r") as file:
            self.genre_name = json.load(file)  
        
        return_dict = {"foot_1s_score": foot_1s_score, "hand_1s_score": hand_1s_score, 
                       "foot_2s_score": foot_2s_score, "hand_2s_score": hand_2s_score, 
                       "genre_Tcount": self.genre_Tcount, "genre_name": self.genre_name, "total": self.total}
          
        return return_dict

    def calc_hits(self, foot_score, hand_score):

        foot_hits = set(foot_score)
        hand_hits = set(hand_score)
        combined_hits = foot_hits.union(hand_hits)
        common_hits = foot_hits.intersection(hand_hits)
        foot_only_hits = len(foot_hits)-len(common_hits)
        hand_only_hits = len(hand_hits)-len(common_hits)
        
        print("total:", self.total)
        print("foot hits:", len(foot_hits), f"({round(len(foot_hits)*100/self.total, 2)} %)" )
        print("hand hits:", len(hand_hits), f"({round(len(hand_hits)*100/self.total, 2)} %)" )
        print("combined hits:", len(combined_hits), f"({round(len(combined_hits)*100/self.total, 2)} %)" )
        print("common hits:", len(common_hits))
        print("foot - common:", foot_only_hits)
        print("hand - common:", hand_only_hits)
        
        return_dict = {"foot_hits": foot_hits, "hand_hits": hand_hits, "combined_hits": combined_hits,
                       "common_hits": common_hits, "foot_only_hits": foot_only_hits, "hand_only_hits": hand_only_hits}
        
        return return_dict


    def calc_totalhits(self, hits_1s, hits_2s):

        # Define hit counts
        hit_counts = {
            "total": self.total,
            "foot1S_hits": len(hits_1s["foot_hits"]),
            "hand1S_hits": len(hits_1s["hand_hits"]),
            "combined1S_hits": len(hits_1s["combined_hits"]),
            "common1S_hits": len(hits_1s["common_hits"]),
            # "foot_only1S_hits": foot_only1S_hits,   # foot - common hits
            # "hand_only1S_hits": hand_only1S_hits,   # hand - common hits
            
            "foot2S_hits": len(hits_2s["foot_hits"]),
            "hand2S_hits": len(hits_2s["hand_hits"]),
            "combined2S_hits": len(hits_2s["combined_hits"]),
            "common2S_hits": len(hits_2s["common_hits"]),
            # "foot_only2S_hits": (foot_only2S_hits),
            # "hand_only2S_hits": hand_only2S_hits,
        }

        hit_data = [{"hit_type": key, "hit": value} for key, value in hit_counts.items() if key != "total"]
        hit_total = pd.DataFrame(hit_data)
        hit_total["percentage"] = round((hit_total["hit"] / hit_counts["total"]) * 100, 2)
        hit_total["method"] = hit_total["hit_type"].apply(lambda x: "1S" if "1S" in x else "2S")

        # Add 'label' column based on 'hit_type'
        hit_total["label"] = hit_total["hit_type"].apply(
            lambda x: f"{self.side_foot} Foot" if x == "foot1S_hits" else
                    f"{self.side_hand} Hand" if x == "hand1S_hits" else
                    f"{self.side_hand} Hand+{self.side_foot} Foot" if x == "combined1S_hits" else
                    "Mutual" if x == "common1S_hits" else
                    "Both Feet" if x == "foot2S_hits" else
                    "Both Hand" if x == "hand2S_hits" else
                    "Both Hand+Foot" if x == "combined2S_hits" else
                    "Mutual"  # For common2S_hits
        )
        # Reorder columns
        hit_total = hit_total[["hit_type", "label", "hit", "percentage", "method"]]
        hit_total.to_csv(f"./stats/{self.norm_mode}/tempo_{self.a}_{self.b}/{self.mode}/hits_total/{self.hand1s_name}_{self.foot1s_name}.csv", index=False)
        
    def calc_genrewise_hits(self, hits_1s, hits_2s):
        
        config_hits = {
            "combined_1S_2S": {
                "L1": [hits_1s["combined_hits"], hits_2s["combined_hits"]],
                "para": ["combined_hits_1S", "combined_hits_2S", f"./stats/{self.norm_mode}/tempo_{self.a}_{self.b}/{self.mode}/genre_wise/combined_{self.hand1s_name}_{self.foot1s_name}_genrewise_hits"]
            },
            "hand_1S": {
                "L1": [hits_1s["hand_hits"], hits_1s["foot_hits"]],
                "para": ["hand_hits_1S", "foot_hits_1S", f"./stats/{self.norm_mode}/tempo_{self.a}_{self.b}/{self.mode}/genre_wise/{self.hand1s_name}_{self.foot1s_name}_1S_genrewise_hits"]
            },
            "hand_2S": {
                "L1": [hits_2s["hand_hits"], hits_2s["foot_hits"]],
                "para": ["hand_hits_2S", "foot_hits_2S", f"./stats/{self.norm_mode}/tempo_{self.a}_{self.b}/{self.mode}/genre_wise/both_hand_foot_2S_genrewise_hits"]
            }
        }

        # Loop through each configuration
        for key, cfg in config_hits.items():
            L1 = cfg["L1"]
            para = cfg["para"]

            column_names = ["Agenre_name", f"A{para[0]}", "Atotal", "Apercentage",
                            "Bgenre_name", f"B{para[1]}", "Btotal", "Bpercentage"]

            df_list = []
            for item in L1:
                hit_idx = list(item)
                hit_df = self.foot_1S_df.iloc[hit_idx]  # Ensure foot_1S_df is defined
                grouped = hit_df.groupby(['dance_genre']).size().reset_index(name='count')

                # Add total and percentage columns
                grouped['total'] = grouped['dance_genre'].map(self.genre_Tcount)
                grouped['percentage'] = round((grouped['count'] / grouped['total']) * 100, 2)
                grouped['genre_name'] = grouped['dance_genre'].map(self.genre_name)

                # Keep only relevant columns
                grouped = grouped[['genre_name', 'count', 'total', 'percentage']]
                df_list.append(grouped)

            # Concatenate and save final dataframe
            final_df = pd.concat(df_list, axis=1)
            final_df.columns = column_names
            final_df.to_csv(para[2] + ".csv", index=False)

            print(f"Saved CSV for {key}: {para[2]}.csv ✅")