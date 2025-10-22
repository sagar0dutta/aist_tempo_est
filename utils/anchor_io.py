import os
import pickle



anchor_base_dir = "./extracted_anchors"
com_base_dir = "./extracted_anchors/com"

com_save_dirs = {
    "hips": "./computed_com/com2d_hips",
    "shoulders": "./computed_com/com2d_shoulders",
    "torso": "./computed_com/com2d_torso"
}

# Create output directories if not exist
for d in com_save_dirs.values():
    os.makedirs(d, exist_ok=True)
    
    

def get_anchor_path(marker, anchor_type, axis, mode, filename, base_dir=anchor_base_dir):
    return f"{base_dir}/{marker}/{anchor_type}/{axis}/{marker}_{mode}_{filename}"

def get_com_path(anchor_type, com_part, axis, mode, filename, base_dir=com_base_dir):
    return f"{base_dir}/{anchor_type}/{com_part}/{axis}/{mode}_{filename}"


def get_all_anchor_paths(anchor_type, mode, filename):
    """
    Returns all marker and COM anchor paths for a given anchor type and mode.

    Parameters
    ----------
    anchor_type : str
        One of ['anchor_zero', 'anchor_peak', 'anchor_energy']
    mode : str
        Either 'uni' or 'bi'
    filename : str
        The base filename (e.g., 'waack_dancer01_cam02_music03_choreo05.pkl')

    Returns
    -------
    dict : containing all relevant file paths structured by body part.
    """

    markers = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    com_parts = ["com_torso", "com_hips", "com_shoulders"]

    data_paths = {
        "markers": {},
        "com": {}
    }

    # Body markers (each with ax0 and ax1)
    for m in markers:
        data_paths["markers"][m] = {
            "ax0": get_anchor_path(m, anchor_type, "ax0", mode, filename),
            "ax1": get_anchor_path(m, anchor_type, "ax1", mode, filename),
            "resultant": get_anchor_path(m, anchor_type, "resultant", mode, filename),
        }

    # COM parts (each with ax0 and ax1)
    for c in com_parts:
        data_paths["com"][c] = {
            "ax0": get_com_path(anchor_type, c, "ax0", mode, filename),
            "ax1": get_com_path(anchor_type, c, "ax1", mode, filename),
        }

    return data_paths

def create_output_dir(main_dir, tempo_dir):
    # 
    directories = [f"{tempo_dir}/anchor_zero", 
                   f"{tempo_dir}/anchor_peak", 
                   f"{tempo_dir}/anchor_energy", 
                   ]
    
    for dir_path in directories:
        full_path = os.path.join(main_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)


def create_anchor_dir(mdir):

    directories = [
                   f"{mdir}/anchor_zero", 
                   f"{mdir}/anchor_zero/ax0",
                   f"{mdir}/anchor_zero/ax1", 
                   f"{mdir}/anchor_zero/resultant",
                   
                   f"{mdir}/anchor_peak", 
                   f"{mdir}/anchor_peak/ax0",
                   f"{mdir}/anchor_peak/ax1", 
                   f"{mdir}/anchor_peak/resultant",
                   
                   f"{mdir}/anchor_energy", 
                   f"{mdir}/anchor_energy/ax0",
                   f"{mdir}/anchor_energy/ax1", 
                   f"{mdir}/anchor_energy/resultant",
                   ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)





def create_com_dir(main_dir):
    os.makedirs(main_dir, exist_ok=True)
    directories = [
                    f"{main_dir}/anchor_zero/com_hips", f"{main_dir}/anchor_zero/com_hips/ax0", f"{main_dir}/anchor_zero/com_hips/ax1",
                    f"{main_dir}/anchor_zero/com_shoulders", f"{main_dir}/anchor_zero/com_shoulders/ax0", f"{main_dir}/anchor_zero/com_shoulders/ax1",
                    f"{main_dir}/anchor_zero/com_torso", f"{main_dir}/anchor_zero/com_torso/ax0", f"{main_dir}/anchor_zero/com_torso/ax1", 
                    
                    f"{main_dir}/anchor_peak/com_hips", f"{main_dir}/anchor_peak/com_hips/ax0", f"{main_dir}/anchor_peak/com_hips/ax1",
                    f"{main_dir}/anchor_peak/com_shoulders", f"{main_dir}/anchor_peak/com_shoulders/ax0", f"{main_dir}/anchor_peak/com_shoulders/ax1",
                    f"{main_dir}/anchor_peak/com_torso", f"{main_dir}/anchor_peak/com_torso/ax0", f"{main_dir}/anchor_peak/com_torso/ax1",
                    
                    f"{main_dir}/anchor_energy/com_hips", f"{main_dir}/anchor_energy/com_hips/ax0", f"{main_dir}/anchor_energy/com_hips/ax1",
                    f"{main_dir}/anchor_energy/com_shoulders", f"{main_dir}/anchor_energy/com_shoulders/ax0", f"{main_dir}/anchor_energy/com_shoulders/ax1",
                    f"{main_dir}/anchor_energy/com_torso", f"{main_dir}/anchor_energy/com_torso/ax0", f"{main_dir}/anchor_energy/com_torso/ax1",
                   ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)



def load_pickle(filepath):
    with open(filepath, "rb") as f:
        json_data = pickle.load(f)
    return json_data

def save_to_pickle(filepath, data):
    # filepath = os.path.join(savepath, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)