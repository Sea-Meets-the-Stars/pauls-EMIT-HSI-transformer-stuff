### Goal:  We want to download a new dataset for pretraining our model.  All we need is hypercubes,
###        ... and don't need to worry about plumes.  So we grab all the L1Bs we can find and
###        ... chip them into 128x128 chips.  We'll save them as .h5's in a new directory. 

### TODO:
# Should match download_plume_dataset.py as much as possible, but can ignore things related to plumes/gas 
# ... since we're grabbing all hypercubes with this one not just ones with plumes
# Should use emit_utils.py as much as possible

# Example usage:
"""
python download_just_hypercubes.py \
  --start_date 2023-06-01 \
  --end_date 2023-06-30 \
  --cloud_cover_min 0 \
  --cloud_cover_max 5 \
  --max_granules 3 \
  --chip_size 128 \
  --output_dir /data/emit_pretrain_test
"""


### Imports
import argparse
import sys
import os
import shutil
import earthaccess
import emit_utils
import xarray
import numpy as np
import gc
import h5py
import time

sys.path.append("/workspace/EMIT-Data-Resources/python/modules")
from emit_tools import emit_xarray

### Function Definitions

#1. Search for granule IDs that match the search criteria.  Will have to be our own custom function 
#   ... since existing functions are based on finding plumes first, 
#   ... but we want a simple search for all hypercubes.
def search_granule_ids(start_date: str, end_date: str, cloud_cover_min: int, cloud_cover_max: int, max_granules: int):
    """
    Search NASA Earthdata for EMIT granule IDs within a given date range and cloud cover range.
    """
    date_range = (start_date, end_date)
    cloud_cover_range = (cloud_cover_min, cloud_cover_max)
    
    earthaccess.login(persist=True)
    granule_metas = earthaccess.search_data(
        short_name="EMITL1BRAD",
        temporal=date_range,
        cloud_cover=cloud_cover_range,
        count=max_granules,
    )
    
    #Get granule names from granule_ids:
    granule_names = [r["umm"]["GranuleUR"].split("_")[4] for r in granule_metas]
    return granule_metas, granule_names

#2. Use emit_utils.download_one_l1b_hypercube to download the hypercube.  Use run_mag1c=False
#   ... and save it to the output directory.  This also orthorectifies the hypercube.
def download_hypercube(l1b_meta, temp_dir):
    """
    Download a given EMIT granule from NASA Earthdata and orthorectify the hypercube.
    """
    earthaccess.login(persist=True)
    downloaded_paths = earthaccess.download(l1b_meta, temp_dir, show_progress=False)
    rad_path = None
    obs_path = None
    
    for path in downloaded_paths:
        if "RAD" in os.path.basename(path):
            rad_path = path
        elif "OBS" in os.path.basename(path):
            obs_path = path
    
    if rad_path is None:
        raise ValueError("No RAD file found in downloaded paths")
    # if obs_path is None:
    #     raise ValueError("No OBS file found in downloaded paths")
    
    hypercube = emit_xarray(rad_path, ortho=True).load()
    #metadata = something with the obs_path, for later
    
    np_hypercube = np.asarray(hypercube.radiance)
    
    # Clear temp dir:
    for path in downloaded_paths:
        os.remove(path)
    gc.collect()
    
    return np_hypercube
    


#3. Chip hypercube into 128x128 chips.  This has to be a custom function since we're not using the plume-based chipping function.
def chip_hypercube(hypercube: np.ndarray, chip_size: int) -> list[np.ndarray]:
    """
    Chip a given hypercube into 128x128 chips.
    """
    # First make a map of what can and can't be chipped in the ortho'd hypercube.
    # Take advantage of the ortho fill being -9999.

    # Grab first channel of hypercube as our mask
    invalid_mask = hypercube[:,:,0] == -9999
    
    chips = []
    chip_step = chip_size // 4
    for i in range(0, hypercube.shape[0] - chip_size, chip_step):
        for j in range(0, hypercube.shape[1] - chip_size, chip_step):
            mask_to_chip = np.zeros((invalid_mask.shape[0], invalid_mask.shape[1]), dtype=bool)
            mask_to_chip[i:i+chip_size, j:j+chip_size] = 1

            chip_safety = sum(invalid_mask[mask_to_chip])
            #print(chip_safety)
            if chip_safety > 0:
                #Chip is unsafe, skip
                continue
            else:
                #Chip is safe, chip it
                chip = hypercube[i:i+chip_size, j:j+chip_size, :]
                chips.append(chip)
                invalid_mask[i:i+chip_size, j:j+chip_size] = 1
    return chips
    
    
#4. Save chips as .h5's.
def save_chips(chips: list[np.ndarray], output_dir: str):
    """
    Save chips as .h5's.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_paths = []
    for i, chip in enumerate(chips):
        save_path = os.path.join(output_dir, f"chip_{i}.h5")
        save_paths.append(save_path)
        with h5py.File(save_path, "w") as f:
            f.create_dataset("hypercube", data=chip)
    return save_paths

### Arguments
# Should get:
# - Root path for saving
# - Start and end dates
# - Cloud Cover max and min
# - Max number of granules to download
# - Chip size
# - Actually this should just be default behavior: "Whether or not to overwrite existing chips, or whether to check for incomplete downloads

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_just_hypercubes.py",
        description=(
            "Download and chip all hypercubes from NASA EMIT for a given date range and cloud cover range."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # --- Search parameters ---
    parser.add_argument(
        "--start_date",
        type=str,
        default="2023-01-01",
        metavar="YYYY-MM-DD",
        help="Start of the temporal search window.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-12-31",
        metavar="YYYY-MM-DD",
        help="End of the temporal search window.",
    )
    parser.add_argument(
        "--cloud_cover_max",
        type=int,
        default=5,
        help="Maximum cloud cover percentage.",
    )
    parser.add_argument(
        "--cloud_cover_min",
        type=int,
        default=0,
        help="Minimum cloud cover percentage.",
    )
    parser.add_argument(
        "--max_granules",
        type=int,
        default=9999999,
        help="Maximum number of granules to download.",
    )
    parser.add_argument(
        "--chip_size",
        type=int,
        default=128,
        help="Size of the chips to chip.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./emit_hypercubes",
        help="Root directory where the hypercubes will be written.",
    )

    return parser

### Main 
def main():
    # Get arguments
    args = _build_argument_parser().parse_args()
    
    # Prep Directories
    output_root_dir = args.output_dir
    os.makedirs(output_root_dir, exist_ok=True)
    
    temp_dir = os.path.join(output_root_dir, "temp")
    # Clear temp directory if it exists:
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
  
    # Search earthdata for granules
    granule_metas, granule_names = search_granule_ids(args.start_date, args.end_date, args.cloud_cover_min, args.cloud_cover_max, args.max_granules)
    print(f"Found {len(granule_metas)} granules")
    
    total_chips = 0
    pre_completed_granules = 0
    granules_completed = 0
    total_granules_expected = len(granule_metas)
    
    start_time = time.time()
    # For each granule...
    for granule_meta, granule_name in zip(granule_metas, granule_names):
        # Check if the granule has already been downloaded and is complete
        granule_output_dir = os.path.join(output_root_dir, granule_name)
        incomplete_flag_file = os.path.join(granule_output_dir, "incomplete.flag")
        complete_flag_file = os.path.join(granule_output_dir, "complete.flag")
        
        if os.path.exists(granule_output_dir):
            if os.path.exists(incomplete_flag_file) or not os.path.exists(complete_flag_file):
                print(f"Granule {granule_name} exists but is incomplete. Deleting and re-downloading")
                shutil.rmtree(granule_output_dir)
            else:
                print(f"Granule {granule_name} exists and is complete. Skipping")
                pre_completed_granules += 1
                continue

        print(f"Granule {granule_name} does not exist. Downloading")
        os.makedirs(granule_output_dir, exist_ok=True)
        with open(incomplete_flag_file, "w") as f:
            f.write("In progress")
        
        # Download and orthorectify the granule
        hypercube = download_hypercube(granule_meta, temp_dir)

        # Chip the hypercube
        chips = chip_hypercube(hypercube, args.chip_size)

        # Save chips as .h5's
        save_chips(chips, granule_output_dir)
        
        # Verify completeness:
        num_chips_saved = len([f for f in os.listdir(granule_output_dir) if f.endswith(".h5")])
        if num_chips_saved == len(chips):
            print(f"Granule {granule_name} completed. {len(chips)} chips saved.")
            os.remove(incomplete_flag_file)
            with open(complete_flag_file, "w") as f:
                f.write("Complete")
            total_chips += len(chips)
        else:
            print(f"WARNING:Granule {granule_name} is incomplete. {num_chips_saved} chips saved out of {len(chips)}.")
            continue
        
        granules_completed += 1
        elapsed_time = time.time() - start_time
        eta_seconds = elapsed_time / granules_completed * (total_granules_expected - granules_completed - pre_completed_granules)
        h = int(eta_seconds // 3600)
        m = int((eta_seconds % 3600) // 60)
        s = int(eta_seconds % 60)
        print(f"Granules Completed: {granules_completed + pre_completed_granules}/{total_granules_expected} | Estimated time remaining: {h}h {m}m {s}s")
                        
    # Report final stats
    print(f"Total granules processed: {len(granule_metas)}")
    print(f"Total chips saved: {total_chips}")
if __name__ == "__main__":
    main()