##Imports
import sys, os, gc
import numpy as np
import pandas as pd
import json
import shutil

import requests
from PIL import Image
from io import BytesIO
import tempfile
import h5py

import earthaccess
from georeader.readers import emit
import geopandas
import rioxarray
from scipy.ndimage import gaussian_filter

from rasterio.enums import Resampling
import matplotlib.pyplot as plt

sys.path.append('/workspace/STARCOP/starcop')
from starcop.models import mag1c_emit

sys.path.append("/workspace/EMIT-Data-Resources/python/modules")
from emit_tools import emit_xarray


KNOWN_BAD_GRANULE_IDS = ["20240812T213914"]

#Quantile thresholds, these are the Q25, Q50, Q75 thresholds for each of the four tracked metrics.  Generated against 100 random plumes in scratch.ipynb:
DIFFICULTY_QUARTILES = {
    "ch4": {
        "max_signal_to_uncertainty": [5.35, 6.64, 10.62],
        "mean_plume_sensitivity": [0.95, 0.99, 1.04],
        "plume_pixel_area": [288.5, 620, 2280],
        "background_std_dev": [157.0, 215.7, 343.2],
    },
    "co2": {
        "max_signal_to_uncertainty": [5.48, 6.78, 8.46],
        "mean_plume_sensitivity": [0.88, 1.04, 1.13],
        "plume_pixel_area": [546.75, 1366.5, 3501.75],
        "background_std_dev": [9363.35, 11912.3, 14599],
    }
}

def get_plume_granule_ids(gas_type="CH4", date_range=('2022-08-01', '2026-12-31'), max_count=100, cloud_cover_range=(0, 5)):
    """
    Searches NASA Earthdata for EMIT plume complexes and returns a list of unique Granule IDs.
    gas_type can be "CH4", "CO2", "either", or "both".
    """
    earthaccess.login(persist=True)
    gas_type = gas_type.lower()
    
    # Map the requested gas to the correct NASA DAAC short names
    if gas_type == "ch4":
        short_names = ["EMITL2BCH4PLM"]
    elif gas_type == "co2":
        short_names = ["EMITL2BCO2PLM"]
    elif gas_type in ["either", "both"]:
        short_names = ["EMITL2BCH4PLM", "EMITL2BCO2PLM"]
    else:
        raise ValueError("gas_type must be 'CH4', 'CO2', 'either', or 'both'")
        
    print(f"Searching for '{gas_type}' plumes between {date_range[0]} and {date_range[1]}...")
    
    plume_results = earthaccess.search_data(
        short_name=short_names,
        temporal=date_range,
        cloud_cover=cloud_cover_range,
        count=max_count,
        version="002",
    )
    
    # Track IDs in separate sets to allow for intersection/union logic
    ch4_ids = set()
    co2_ids = set()
    
    for result in plume_results:
        granule_ur = result['umm']['GranuleUR']
        # Extract the timestamp ID (e.g., '20230825T163454')
        granule_id = granule_ur.split('_')[4]
        
        # Check the UR string directly for the gas type
        if "CH4" in granule_ur:
            ch4_ids.add(granule_id)
        elif "CO2" in granule_ur:
            co2_ids.add(granule_id)
            
    # Apply the requested filtering logic
    if gas_type == "ch4":
        final_ids = ch4_ids
    elif gas_type == "co2":
        final_ids = co2_ids
    elif gas_type == "either":
        final_ids = ch4_ids.union(co2_ids)
    elif gas_type == "both":
        final_ids = ch4_ids.intersection(co2_ids)
        
    granule_id_list = list(final_ids)
    print(f"Returned {len(granule_id_list)} unique Granule IDs for gas_type '{gas_type}'.")
    
    return granule_id_list

def get_all_granule_products(granule_id):
    """
    Searches NASA Earthdata for all EMIT products associated with a specific Granule ID.
    Returns a dictionary grouping the raw Earthdata results by product type.
    """
    earthaccess.login(persist=True)
    print(f"Querying Earthdata for all EMIT products linked to Granule ID: {granule_id}...")
    
    # NASA CMR requires us to specify the collections to avoid global wildcard searches.
    # We will provide a list of all major EMIT short names.
    emit_short_names = [
        "EMITL1BRAD",    # L1B Radiance
        "EMITL1BOBS",    # L1B Observation Data
        "EMITL2ARFL",    # L2A Reflectance
        "EMITL2AMASK",   # L2A Mask
        "EMITL2BCH4ENH", # L2B CH4 Enhancements
        "EMITL2BCH4PLM", # L2B CH4 Plumes
        "EMITL2BCO2ENH", # L2B CO2 Enhancements
        "EMITL2BCO2PLM", # L2B CO2 Plumes
        "EMITL2BMIN"     # L2B Minerals
    ]
    
    results = earthaccess.search_data(
        short_name=emit_short_names,
        granule_name=f"*{granule_id}*"
    )
    
    products_dict = {}
    
    for result in results:
        granule_ur = result['umm']['GranuleUR']
        
        # Parse the product type directly from the Granule UR 
        # Example: EMIT_L1B_RAD_001_20230825T163454_... -> "L1B_RAD"
        # Example: EMIT_L2B_CH4PLM_001_... -> "L2B_CH4PLM"
        parts = granule_ur.split('_')
        if len(parts) >= 3:
            product_key = f"{parts[1]}_{parts[2]}"
        else:
            product_key = "UNKNOWN_PRODUCT"
            
        if product_key not in products_dict:
            products_dict[product_key] = []
            
        products_dict[product_key].append(result)
        
    print(f"Found {len(results)} total files across {len(products_dict)} product categories:")
    for key, items in products_dict.items():
        print(f"  - {key}: {len(items)} file(s)")
        
    return products_dict

def display_plume_previews(granule_id, gas_type="ch4"):
    """
    Downloads and displays EMIT browse images (previews) for a given granule ID.
    Figure 1 is L1B Radiance. Figures 2/3 are L2B Enhancements (v2).
    gas_type can be 'ch4', 'co2', or 'both'.
    """
    earthaccess.login(persist=True)
    gas_type = gas_type.lower()
    
    # Map the requested gas to the correct NASA DAAC short names
    short_names = ["EMITL1BRAD"]
    if gas_type in ["ch4", "both"]:
        short_names.append("EMITL2BCH4ENH")
    if gas_type in ["co2", "both"]:
        short_names.append("EMITL2BCO2ENH")
        
    print(f"Fetching browse file metadata for {granule_id}...")
    results = earthaccess.search_data(
        short_name=short_names,
        granule_name=f"*{granule_id}*"
    )
    
    # Extract the browse URLs (dataviz_links)
    browse_urls = {}
    for result in results:
        short_name = result['umm']['CollectionReference']['ShortName']
        version = result['umm']['CollectionReference']['Version']
        
        # Ensure we only grab Version 2 for the L2B enhancements
        if "L2B" in short_name and version != "002":
            continue
            
        viz_links = result.dataviz_links()
        if viz_links:
            browse_urls[short_name] = viz_links[0]
            
    if not browse_urls:
        print("No browse images found for this granule.")
        return
        
    def load_image(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
        
    print("Downloading and rendering browse images...")
    
    # Set up layout: 1x3 if both, else 1x2
    num_figs = 3 if gas_type == "both" else 2
    fig, axes = plt.subplots(1, num_figs, figsize=(8 * num_figs, 8))
    
    # Figure 1: L1B Preview
    axes[0].set_title(f"Figure 1: L1B Radiance\nGranule: {granule_id}")
    axes[0].axis('off')
    if "EMITL1BRAD" in browse_urls:
        axes[0].imshow(load_image(browse_urls["EMITL1BRAD"]))
    else:
        axes[0].text(0.5, 0.5, 'L1B Browse Not Found', ha='center', va='center')

    # Figure 2 & 3: Enhancements
    if gas_type in ["ch4", "both"]:
        axes[1].set_title("Figure 2: L2B CH4 Enhancement (v2)")
        axes[1].axis('off')
        if "EMITL2BCH4ENH" in browse_urls:
            axes[1].imshow(load_image(browse_urls["EMITL2BCH4ENH"]))
        else:
            axes[1].text(0.5, 0.5, 'CH4 ENH Browse Not Found', ha='center', va='center')
            
    if gas_type == "co2":
        axes[1].set_title("Figure 2: L2B CO2 Enhancement (v2)")
        axes[1].axis('off')
        if "EMITL2BCO2ENH" in browse_urls:
            axes[1].imshow(load_image(browse_urls["EMITL2BCO2ENH"]))
        else:
            axes[1].text(0.5, 0.5, 'CO2 ENH Browse Not Found', ha='center', va='center')
            
    elif gas_type == "both":
        axes[2].set_title("Figure 3: L2B CO2 Enhancement (v2)")
        axes[2].axis('off')
        if "EMITL2BCO2ENH" in browse_urls:
            axes[2].imshow(load_image(browse_urls["EMITL2BCO2ENH"]))
        else:
            axes[2].text(0.5, 0.5, 'CO2 ENH Browse Not Found', ha='center', va='center')

    plt.tight_layout()
    plt.show()
    
## Get list of L2B products:
def search_l2b_metadata(date_range=('2023-01-01', '2023-12-31'), max_count=10, cloudcover_max=1, cloudcover_min=0, gas_type="ch4"):
    gas_type = gas_type.lower()
    if gas_type == "ch4":
        short_names = ["EMITL2BCH4PLM"]
    elif gas_type == "co2":
        short_names = ["EMITL2BCO2PLM"]
    elif gas_type in ["either", "both"]:
        short_names = ["EMITL2BCH4PLM", "EMITL2BCO2PLM"]
    else:
        raise ValueError("gas_type must be 'ch4', 'co2', 'either', or 'both'")

    earthaccess.login(persist=True)

    print(f"Searching for {gas_type} L2B Plume data in date range: {date_range}...")
    l2b_plume_results = earthaccess.search_data(
        short_name=short_names,
        temporal=date_range,
        cloud_cover=(cloudcover_min, cloudcover_max),
        count=max_count,
        version="002",
    )
    print(f"Found {len(l2b_plume_results)} L2B Plume products")

    return l2b_plume_results

## Get unique granule IDs:
def get_unique_granule_ids(l2b_metadata_list):
    granule_ids = []
    for l2b_product in l2b_metadata_list:
        granule_id = l2b_product['umm']['GranuleUR']
        granule_id = granule_id.split('_')[4]
        granule_ids.append(granule_id)
    
    granule_ids = list(set(granule_ids))
    print(f"Found {len(granule_ids)} unique granule IDs")
    return granule_ids

## Get list of L1B products:
def get_l1b_metadata(granule_ids):
    earthaccess.login(persist=True)
    l1b_products = []
    for granule_id in granule_ids:
        l1b_products.extend(earthaccess.search_data(
            short_name="EMITL1BRAD",
            granule_name=f'*{granule_id}*',
        ))
    return l1b_products

## Download hypercube + run MAG1C and get metadata:
def download_one_l1b_hypercube(granule_id, temp_dir="/more_data/temp_processing/emit_download/", 
                               show_progress=False, run_mag1c=True):
    earthaccess.login(persist=True)
    
    l1b_metadata = get_l1b_metadata([granule_id])
    
    os.makedirs(temp_dir, exist_ok=True)
    downloaded_paths = earthaccess.download(l1b_metadata, temp_dir, show_progress=show_progress)
    
    rad_path = None
    for path in downloaded_paths:
        if "RAD" in os.path.basename(path):
            rad_path = path
            break
    
    if rad_path is None:
        raise ValueError("No RAD file found in downloaded paths")
    print(rad_path)
    
    mag1c_output = None
    if run_mag1c:
        rst = emit.EMITImage(rad_path)
        mag1c_output, _ = mag1c_emit.mag1c_emit(rst, column_step=2, georreferenced=True)
        del rst
    
    l1b_product = emit_xarray(rad_path, ortho=True).load()
        
    #remove files in temp_dir:
    for path in downloaded_paths:
        os.remove(path)
    
    gc.collect()
    return l1b_product, mag1c_output


def get_l2bs_for_granule_id(granule_id, gas_type="ch4"):
    """
    granule_id: e.g. "20230107T143818" (the timestamp you extracted)
    gas_type: "ch4" or "co2"
    Returns the plume complex and enhancement search results for the specified gas.
    """
    earthaccess.login(persist=True)
    gas_type = gas_type.lower()

    if gas_type == "ch4":
        plm_short_name = "EMITL2BCH4PLM"
        enh_short_name = "EMITL2BCH4ENH"
    elif gas_type == "co2":
        plm_short_name = "EMITL2BCO2PLM"
        enh_short_name = "EMITL2BCO2ENH"
    else:
        raise ValueError(f"gas_type must be 'ch4' or 'co2', got '{gas_type}'")

    plume_granules = earthaccess.search_data(
        short_name=plm_short_name, version="002", granule_name=f"*{granule_id}*"
    )
    enh_layers = earthaccess.search_data(
        short_name=enh_short_name, version="002", granule_name=f"*{granule_id}*"
    )

    return plume_granules, enh_layers


def download_one_l2b_set(granule_id, gas_type="ch4", temp_dir="/more_data/temp_processing/emit_download/", 
                         onlytifs=False, show_progress=False, nasa_archive_dir=None):
    l2b_plm_metas, l2b_enh_metas = get_l2bs_for_granule_id(granule_id, gas_type=gas_type)
    os.makedirs(temp_dir, exist_ok=True)
    downloaded_paths = earthaccess.download(l2b_plm_metas, temp_dir, show_progress=show_progress)
    l2b_jsons = []
    l2b_tifs = []
    for path in downloaded_paths:
        if os.path.basename(path).lower().endswith(".json") and not onlytifs:
            plume_metadata = geopandas.read_file(path)
            l2b_jsons.append(plume_metadata)
        elif os.path.basename(path).lower().endswith(".tif"):
            plume_mask = (
                rioxarray.open_rasterio(path).squeeze("band", drop=True).load()
            )  # load now: temp dir may be moved/removed before later .values access
            l2b_tifs.append(plume_mask)
            
    if nasa_archive_dir is not None:
        os.makedirs(nasa_archive_dir, exist_ok=True)
        for path in downloaded_paths:
            shutil.move(path, os.path.join(nasa_archive_dir, os.path.basename(path)))
    else:
        for path in downloaded_paths:
            os.remove(path)
    
    l2b_enhs = []
    l2b_uncerts = []
    l2b_sens = []
    if not onlytifs:
        downloaded_paths = earthaccess.download(l2b_enh_metas, temp_dir, show_progress=show_progress)
        for path in downloaded_paths:
            basename = os.path.basename(path)
            if not basename.lower().endswith(".tif"):
                continue
            enh_tag = "CH4ENH" if gas_type.lower() == "ch4" else "CO2ENH"
            uncert_tag = "CH4UNCERT" if gas_type.lower() == "ch4" else "CO2UNCERT"
            sens_tag = "CH4SENS" if gas_type.lower() == "ch4" else "CO2SENS"
            if enh_tag in basename:
                l2b_enhs.append(rioxarray.open_rasterio(path).squeeze("band", drop=True).load())
            elif uncert_tag in basename:
                l2b_uncerts.append(rioxarray.open_rasterio(path).squeeze("band", drop=True).load())
            elif sens_tag in basename:
                l2b_sens.append(rioxarray.open_rasterio(path).squeeze("band", drop=True).load())
                
        if nasa_archive_dir is not None:
            for path in downloaded_paths:
                shutil.move(path, os.path.join(nasa_archive_dir, os.path.basename(path)))
        else:
            for path in downloaded_paths:
                os.remove(path)
                
    gc.collect()
    return l2b_jsons, l2b_tifs, l2b_enhs, l2b_uncerts, l2b_sens

def _crop_to_plume_vicinity(plume_mask, enh_array, uncert_array, sens_array, margin):
    """
    Crop all arrays to the axis-aligned bounding box of plume pixels plus margin
    (clipped to array edges). Used so background statistics reflect local clutter
    near the plume, not the full ~75 km granule.
    """
    margin = int(margin)
    pm = plume_mask > 0
    if not np.any(pm):
        return plume_mask, enh_array, uncert_array, sens_array
    rows = np.where(np.any(pm, axis=1))[0]
    cols = np.where(np.any(pm, axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    H, W = plume_mask.shape[0], plume_mask.shape[1]
    r0 = max(0, r0 - margin)
    r1 = min(H - 1, r1 + margin)
    c0 = max(0, c0 - margin)
    c1 = min(W - 1, c1 + margin)
    sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    return (
        plume_mask[sl],
        enh_array[sl],
        uncert_array[sl],
        sens_array[sl],
    )

def assign_training_category(metrics, gas_type="ch4"):
    """
    Translates the four continuous difficulty metrics into a 0-12 Composite Score,
    then maps that score to a categorical training curriculum using predefined quartiles.
    """
    # 1. Handle missing data (plumes masked out by the EMIT pipeline)
    if pd.isna(metrics.get("max_signal_to_uncertainty")) or pd.isna(metrics.get("mean_plume_sensitivity")):
        return "corrupted"

    q = DIFFICULTY_QUARTILES[gas_type.lower()]
    score = 0

    # 2. Standard Scoring (Higher is easier/better)
    for key in ["max_signal_to_uncertainty", "mean_plume_sensitivity", "plume_pixel_area"]:
        val = metrics.get(key)
        if val >= q[key][2]: score += 3       # >= q75 (4th Quartile)
        elif val >= q[key][1]: score += 2     # >= q50 (3rd Quartile)
        elif val >= q[key][0]: score += 1     # >= q25 (2nd Quartile)
        # Values below q25 (1st Quartile) get 0 points

    # 3. Inverted Scoring (Higher clutter is harder/worse)
    bg_val = metrics.get("background_std_dev")
    if pd.isna(bg_val):
        score += 1
        print(f"Background std dev is NaN, adding 1 point to score")
    else:
        if bg_val <= q["background_std_dev"][0]: score += 3      # <= q25 (1st Quartile - Cleanest)
        elif bg_val <= q["background_std_dev"][1]: score += 2    # <= q50 (2nd Quartile)
        elif bg_val <= q["background_std_dev"][2]: score += 1    # <= q75 (3rd Quartile)
        # Values above q75 (4th Quartile - Messiest) get 0 points

    # 4. Map the 0-12 score to the training category
    assert score >= 0 and score <= 12, "Score must be between 0 and 12"
    
    if score <= 3:
        return "very_hard"
    elif score <= 6:
        return "hard"
    elif score <= 9:
        return "medium"
    else:
        return "easy"

def compute_plume_metrics(
    enh_array,
    uncert_array,
    sens_array,
    plume_mask,
    local_crop_margin=128,
):
    """
    Four interpretable difficulty metrics (aligned ENH / UNCERT / SENS / mask).

    Parameters
    ----------
    enh_array, uncert_array, sens_array : array_like
        L2B enhancement (ppm·m), uncertainty (ppm·m), sensitivity (unitless).
    plume_mask : array_like
        Plume mask: positive = plume. Same 2D shape as ENH.
    local_crop_margin : int or None
        If int, crop to plume bounding box plus margin before metrics (local
        background for background_std_dev). plume_pixel_area uses the full
        granule mask count before crop. If None, full-scene background (legacy).

    Returns
    -------
    dict
        max_signal_to_uncertainty, mean_plume_sensitivity, plume_pixel_area,
        background_std_dev (float or NaN where undefined).
    """
    plume_mask = np.asarray(plume_mask)
    enh_array = np.asarray(enh_array, dtype=np.float64)
    uncert_array = np.asarray(uncert_array, dtype=np.float64)
    sens_array = np.asarray(sens_array, dtype=np.float64)

    fill = -9999.0
    enh_array = np.where(enh_array == fill, np.nan, enh_array)
    uncert_array = np.where(uncert_array == fill, np.nan, uncert_array)
    sens_array = np.where(sens_array == fill, np.nan, sens_array)

    plume_pixel_area = float(np.sum(plume_mask > 0))

    if local_crop_margin is not None and plume_pixel_area > 0:
        plume_mask, enh_array, uncert_array, sens_array = _crop_to_plume_vicinity(
            plume_mask, enh_array, uncert_array, sens_array, local_crop_margin
        )

    plume_bool = (plume_mask > 0) & np.isfinite(plume_mask)
    background_bool = ~plume_bool
    n_plume = np.sum(plume_bool)
    n_background = np.sum(background_bool)

    out = {}

    # 1. Max ENH/UNCERT on plume (signal-to-uncertainty)
    if n_plume == 0:
        out["max_signal_to_uncertainty"] = np.nan
        print(f"No plume pixels found in the array.")
        print(f"Plume Area (Pixels): {n_plume}")
        print(f"Valid ENH Pixels:    {np.sum(np.isfinite(enh_array[plume_bool]))}")
        print(f"Valid UNCERT Pixels: {np.sum(np.isfinite(uncert_array[plume_bool]))}")
        print(f"Valid SENS Pixels:   0")
        print(f"Cause: Plume mask falls entirely on -9999 nodata pixels.")
        print(f"--------------------------------------------------\n")
    else:
        enh_plume = enh_array[plume_bool]
        uncert_plume = uncert_array[plume_bool]
        valid = np.isfinite(enh_plume) & np.isfinite(uncert_plume) & (uncert_plume > 0)
        if np.any(valid):
            sur = np.where(valid, enh_plume / uncert_plume, np.nan)
            out["max_signal_to_uncertainty"] = np.nanmax(sur)
        else:
            out["max_signal_to_uncertainty"] = np.nan
            print(f"No valid plume pixels found in the array.")
            print(f"Plume Area (Pixels): {n_plume}")
            print(f"Valid ENH Pixels:    {np.sum(np.isfinite(enh_array[plume_bool]))}")
            print(f"Valid UNCERT Pixels: {np.sum(np.isfinite(uncert_array[plume_bool]))}")
            print(f"Valid SENS Pixels:   0")
            print(f"Cause: Plume mask falls entirely on -9999 nodata pixels.")
            print(f"--------------------------------------------------\n")

    # 2. Mean SENS on plume
    if n_plume == 0:
        out["mean_plume_sensitivity"] = np.nan
    else:
        sens_plume = sens_array[plume_bool]
        valid_sens = np.isfinite(sens_plume)
        
        if not np.any(valid_sens):
            out["mean_plume_sensitivity"] = np.nan
            
            # Print diagnostic information for the user
            print(f"\n--- All-NaN SENS Array Detected ---")
            print(f"Plume Area (Pixels): {n_plume}")
            print(f"Valid ENH Pixels:    {np.sum(np.isfinite(enh_array[plume_bool]))}")
            print(f"Valid UNCERT Pixels: {np.sum(np.isfinite(uncert_array[plume_bool]))}")
            print(f"Valid SENS Pixels:   0")
            print(f"Cause: Plume mask falls entirely on -9999 nodata pixels.")
            print(f"--------------------------------------------------\n")
        else:
            out["mean_plume_sensitivity"] = np.nanmean(sens_plume)

    # 3. Plume pixel count (full granule, before local crop)
    out["plume_pixel_area"] = plume_pixel_area

    # 4. Std dev of ENH on local background
    if n_background == 0:
        out["background_std_dev"] = np.nan
        print(f"No background pixels found in the array.")
        print(f"Plume Area (Pixels): {n_plume}")
        print(f"Valid ENH Pixels:    {np.sum(np.isfinite(enh_array[plume_bool]))}")
        print(f"Valid UNCERT Pixels: {np.sum(np.isfinite(uncert_array[plume_bool]))}")
        print(f"Valid SENS Pixels:   0")
        print(f"Cause: Plume mask falls entirely on -9999 nodata pixels.")
        print(f"--------------------------------------------------\n")
    else:
        enh_bg = enh_array[background_bool]
        finite = np.isfinite(enh_bg)
        if np.sum(finite) < 2:
            out["background_std_dev"] = np.nan
            print(f"No valid background pixels found in the array.")
            print(f"Plume Area (Pixels): {n_plume}")
            print(f"Valid ENH Pixels:    {np.sum(np.isfinite(enh_array[plume_bool]))}")
            print(f"Valid UNCERT Pixels: {np.sum(np.isfinite(uncert_array[plume_bool]))}")
            print(f"Valid SENS Pixels:   0")
            print(f"Cause: Plume mask falls entirely on -9999 nodata pixels.")
            print(f"--------------------------------------------------\n")
        else:
            out["background_std_dev"] = np.nanstd(enh_bg)

    return out

def get_plume_mask_and_stats(plume_tif, visualize_flag=False, gas_type="ch4"):
    plume_mask = plume_tif.values.copy()
    plume_mask[plume_mask == -9999] = np.nan
    
    #calculate basic stats:
    max_ppmm = np.nanmax(plume_mask)
    mean_ppmm = np.nanmean(plume_mask[plume_mask > 0])
    sum_ppmm = np.nansum(plume_mask[plume_mask > 0])
    min_ppmm = np.nanmin(plume_mask)
    
    brightest_99prc = np.nanpercentile(plume_mask[plume_mask > 0], 99)
    num_plume_pixels = np.count_nonzero(plume_mask > 0)
     
    if visualize_flag:
        plt.hist(plume_mask.flatten(), bins=50)
        plt.show()
    
    plume_mask[np.isnan(plume_mask)] = 0
    
    if visualize_flag:
        plt.imshow(plume_mask)
        plt.show()
        
    plume_stats_dict = {
        "max_ppmm": max_ppmm.item(),
        "mean_ppmm": mean_ppmm.item(),
        "sum_ppmm": sum_ppmm.item(),
        "num_plume_pixels": num_plume_pixels,
        "peak_intensity": brightest_99prc.item()
    }
    
    return plume_mask, plume_stats_dict


def project_plume_mask(plume_tif, l1b_product):
    ref = l1b_product["radiance"].isel(wavelengths=0)
    ref = ref.rename({d: {"longitude": "x", "latitude": "y"}[d] for d in ref.dims if d in ("longitude", "latitude")})

    # Ensure CRS exists (EMIT ortho products are lon/lat)
    if ref.rio.crs is None:
        ref = ref.rio.write_crs("EPSG:4326", inplace=False)
    if plume_tif.rio.crs is None:
        plume_tif = plume_tif.rio.write_crs(ref.rio.crs, inplace=False)
        
    plume_out = plume_tif.rio.reproject_match(ref, resampling=Resampling.nearest, nodata=np.nan)
    plume_out.values[plume_out.values == -9999] = np.nan
    
    gc.collect()
    return plume_out


def align_plume_to_enh_grid(plume_tif, l2b_enh, l2b_uncert, l2b_sens):
    """
    Reproject the plume mask raster to the ENH grid and return aligned numpy arrays
    suitable for compute_plume_metrics(enh, uncert, sens, plume_mask).

    Parameters
    ----------
    plume_tif : rioxarray.DataArray
        Plume mask TIF (may use -9999 nodata; plume pixels are > 0).
    l2b_enh, l2b_uncert, l2b_sens : rioxarray.DataArray
        Enhancement, uncertainty, and sensitivity rasters on a common grid.

    Returns
    -------
    plume_mask_binary : np.ndarray
        1 = plume, 0 = background. Same shape as ENH.
    enh_arr, uncert_arr, sens_arr : np.ndarray
        float64 arrays on the ENH grid, same shape as plume_mask_binary.
    """
    ref = l2b_enh
    if ref.rio.crs is None:
        ref = ref.rio.write_crs("EPSG:4326", inplace=False)
    if plume_tif.rio.crs is None:
        plume_tif = plume_tif.rio.write_crs(ref.rio.crs, inplace=False)

    plume_reproj = plume_tif.rio.reproject_match(
        ref, resampling=Resampling.nearest, nodata=np.nan
    )
    plume_vals = plume_reproj.values.copy()
    plume_vals[plume_vals == -9999] = np.nan
    plume_vals[np.isnan(plume_vals)] = 0
    plume_mask_binary = (plume_vals > 0).astype(np.int64)

    enh_arr = np.asarray(l2b_enh.values, dtype=np.float64)
    uncert_arr = np.asarray(l2b_uncert.values, dtype=np.float64)
    sens_arr = np.asarray(l2b_sens.values, dtype=np.float64)
    
    enh_arr[enh_arr == -9999] = np.nan
    uncert_arr[uncert_arr == -9999] = np.nan
    sens_arr[sens_arr == -9999] = np.nan

    return plume_mask_binary, enh_arr, uncert_arr, sens_arr


def chip_plume(l1b_product, plume_mask, l2b_enh, mag1c_layer=None, chip_size=256, step_size=16):
    chip_size = int(chip_size)
    step_size = int(step_size)
    hypercube = l1b_product["radiance"]
    mag1c_arr = mag1c_layer.values.copy() if mag1c_layer is not None else None
    plume_mask = plume_mask.values.copy()
    l2b_enh = l2b_enh.values.copy()
    
    #Use max intensity as a root:
    max_plume_coords = np.argwhere(plume_mask == np.nanmax(plume_mask))[0]
    #print(max_plume_coords)
    
    ##Find chip which has the highest sum of l2b_enh intensity:
    l2b_blur = gaussian_filter(l2b_enh, sigma=10)
    
    #Define search area:
    img_max_x = l2b_blur.shape[0]
    img_max_y = l2b_blur.shape[1]
    min_x = max(0, max_plume_coords[0] - (chip_size))
    max_x = min(img_max_x, max_plume_coords[0] + (chip_size))
    min_y = max(0, max_plume_coords[1] - (chip_size))
    max_y = min(img_max_y, max_plume_coords[1] + (chip_size))
    
    #Search for chip with max sum:
    sums = []
    indices = []
    for x in range(min_x, max_x-chip_size, step_size):
        for y in range(min_y, max_y-chip_size, step_size):
            chip = l2b_blur[int(x):int(x+chip_size), int(y):int(y+chip_size)]
            s = np.nansum(chip)
            sums.append(s)
            indices.append((x, y))
            
    max_sum_idx = np.argmax(sums)
    max_sum_coords = indices[max_sum_idx]
    
    #Get chip:
    chip_start_x = max_sum_coords[0]
    chip_start_y = max_sum_coords[1]
    
    hypercube_chip = hypercube[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size), :]
    mag1c_chip = mag1c_arr[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size)] if mag1c_arr is not None else None
    l2b_enh_chip = l2b_enh[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size)]
    plume_mask_chip = plume_mask[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size)]
    plume_mask_chip[np.isnan(plume_mask_chip)] = 0
    plume_mask_chip[plume_mask_chip == -9999] = 0
    plume_mask_chip[plume_mask_chip < 0] = 0
    
    chip_plume_center_x = max_plume_coords[0] - chip_start_x
    chip_plume_center_y = max_plume_coords[1] - chip_start_y
    

    gc.collect()
    return hypercube_chip, mag1c_chip, plume_mask_chip, l2b_enh_chip, (chip_plume_center_x, chip_plume_center_y), (chip_start_x, chip_start_y)
    

def chip_negatives(l1b_product, l2b_enh, mag1c_layer=None, chip_size=256, step_size=16, chips_to_generate=1, 
                   existing_positive_chips=None, display_chipping = False):
    
    if existing_positive_chips is None:
        existing_positive_chips = []
        
    chip_size = int(chip_size)
    step_size = int(step_size)
    print("Generating " + str(chips_to_generate) + " negative chips")
    hypercube = l1b_product["radiance"]
    mag1c_arr = mag1c_layer.values.copy() if mag1c_layer is not None else None
    l2b_enh = l2b_enh.values.copy()

    #Penalize chips that intersect with the -9999 invalid edges:
    l2b_blur = l2b_enh.copy()
    l2b_blur[l2b_blur == -9999] = 999999.
    l2b_blur = gaussian_filter(l2b_blur, sigma=10)
    
    hypercube_chip_list = []
    mag1c_chip_list = []
    l2b_enh_chip_list = []
    chip_positions_list = []
    
    #Heavily penalize chips that overlap with positive chips
    if len(existing_positive_chips) > 0:
        for chip in existing_positive_chips:
            chip_start_x, chip_start_y = chip
            #print(chip_start_x, chip_start_y)
            l2b_blur[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size)] = 9999999.
                
    for i in range(chips_to_generate):
        if display_chipping:
            plt.imshow(l2b_blur)
            plt.show()
        
        sums = []
        indices = []
        for x in range(0, l2b_blur.shape[0]-chip_size, step_size):
            for y in range(0, l2b_blur.shape[1]-chip_size, step_size):
                chip = l2b_blur[x:x+chip_size, y:y+chip_size]
                s = np.nansum(chip)
                sums.append(s)
                indices.append((x, y))
                
        min_sum_idx = np.argmin(sums)
        min_sum_coords = indices[min_sum_idx]
        #print(sums)
        if np.min(sums) > 100000000:
            print("No more negative chips can be generated from this granule")
            break
        
        #Get chip:
        chip_start_x = min_sum_coords[0]
        chip_start_y = min_sum_coords[1]
        
        hypercube_chip = hypercube[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size), :]
        mag1c_chip = mag1c_arr[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size)] if mag1c_arr is not None else None
        l2b_enh_chip = l2b_enh[int(chip_start_x):int(chip_start_x+chip_size), int(chip_start_y):int(chip_start_y+chip_size)]
        
        hypercube_chip_list.append(hypercube_chip)
        mag1c_chip_list.append(mag1c_chip)
        l2b_enh_chip_list.append(l2b_enh_chip)
        chip_positions_list.append((chip_start_x, chip_start_y))
        
        l2b_blur[chip_start_x:chip_start_x+chip_size, chip_start_y:chip_start_y+chip_size] = 999999.
    
    gc.collect()
    return hypercube_chip_list, mag1c_chip_list, l2b_enh_chip_list, chip_positions_list
    

def survey_plume_stats(
    max_count=100,
    gas_type="ch4",
    include_difficulty_metrics=False,
    temp_dir=None,
):
    """
    Survey plume statistics across L2B plume products.

    Parameters
    ----------
    max_count, gas_type : passed to search_l2b_metadata.
    include_difficulty_metrics : bool
        If False (default), only downloads plume-mask TIFs (fast). Same as before.
        If True, also downloads ENH/UNCERT/SENS for each granule and adds the four
        difficulty metrics (max_signal_to_uncertainty, mean_plume_sensitivity,
        plume_pixel_area, background_std_dev) per plume row.
    temp_dir : str or None
        When include_difficulty_metrics is True, directory used for Earthdata
        downloads. If None, a temporary directory is created per granule and removed
        after processing that granule.
    """
    plume_results = search_l2b_metadata(
        date_range=("2023-01-01", "2025-12-31"),
        max_count=max_count,
        cloudcover_max=3,
        gas_type=gas_type,
    )
    granule_ids = get_unique_granule_ids(plume_results)

    rows = []
    for granule_id in granule_ids:
        if include_difficulty_metrics:
            td = temp_dir if temp_dir is not None else tempfile.mkdtemp(
                prefix=f"emit_survey_{granule_id}_"
            )
            try:
                (
                    _l2b_jsons,
                    l2b_tifs,
                    l2b_enhs,
                    l2b_uncerts,
                    l2b_sens,
                ) = download_one_l2b_set(
                    granule_id,
                    onlytifs=False,
                    gas_type=gas_type,
                    temp_dir=td,
                )
            finally:
                if temp_dir is None:
                    shutil.rmtree(td, ignore_errors=True)

            have_layers = (
                len(l2b_enhs) > 0
                and len(l2b_uncerts) > 0
                and len(l2b_sens) > 0
            )
            for l2b_tif in l2b_tifs:
                _, plume_stats = get_plume_mask_and_stats(l2b_tif, gas_type=gas_type)
                plume_stats["granule_id"] = granule_id
                plume_stats["gas_type"] = gas_type
                if have_layers:
                    plume_mask_binary, enh_arr, uncert_arr, sens_arr = (
                        align_plume_to_enh_grid(
                            l2b_tif,
                            l2b_enhs[0],
                            l2b_uncerts[0],
                            l2b_sens[0],
                        )
                    )
                    difficulty_metrics = compute_plume_metrics(
                        enh_arr, uncert_arr, sens_arr, plume_mask_binary
                    )
                    for k, v in difficulty_metrics.items():
                        difficulty_metrics[k] = float(v) if np.isscalar(v) else v
                    plume_stats = plume_stats | difficulty_metrics
                rows.append(plume_stats)
        else:
            _, l2b_tifs, _, _, _ = download_one_l2b_set(
                granule_id, onlytifs=True, gas_type=gas_type
            )
            for l2b_tif in l2b_tifs:
                _, plume_stats = get_plume_mask_and_stats(l2b_tif, gas_type=gas_type)
                plume_stats["granule_id"] = granule_id
                plume_stats["gas_type"] = gas_type
                rows.append(plume_stats)

    plume_df = pd.DataFrame(rows)
    plume_df.sort_values(by="max_ppmm", ascending=False, inplace=True)

    return plume_df
    

def collate_granule_metadata(granule_id, l1b_product, num_plumes, gas_type="ch4"):
    radiance = l1b_product["radiance"]
    granule_metadata = {
        "granule_id": granule_id,
        "flight_line": l1b_product.attrs["flight_line"],
        "time_coverage_start": l1b_product.attrs["time_coverage_start"],
        "day_night_flag": l1b_product.attrs["day_night_flag"],
        "crs": l1b_product.attrs["spatial_ref"],
        "geotransform": l1b_product.attrs["geotransform"].tolist(),
        "granule_height": int(radiance.shape[0]),
        "granule_width": int(radiance.shape[1]),
        "num_bands": int(radiance.shape[2]),
        "wavelengths": np.asarray(l1b_product["wavelengths"]).tolist(),
        "fwhm": np.asarray(l1b_product["fwhm"]).tolist(),
        "num_plumes": num_plumes,
        "gas_type": gas_type,
    }
    return granule_metadata


def collate_plume_metadata(
    l2b_jsons,
    l2b_tifs,
    granule_id,
    gas_type="ch4",
    l2b_enh=None,
    l2b_uncert=None,
    l2b_sens=None,
):
    """
    Build per-plume metadata dicts. If l2b_enh, l2b_uncert, and l2b_sens are
    provided (one DataArray each for the granule), also attach the four
    difficulty metrics from compute_plume_metrics after aligning the plume
    mask to the ENH grid.
    """
    plume_metadata_list = []
    have_enh_metrics = (
        l2b_enh is not None and l2b_uncert is not None and l2b_sens is not None
    )

    for l2b_json, l2b_tif in zip(l2b_jsons, l2b_tifs):
        _, plume_stats = get_plume_mask_and_stats(l2b_tif, gas_type=gas_type)
        plume_metadata = {
            "plume_id": l2b_json["Plume ID"][0],
            "granule_id": granule_id,
            "gas_type": gas_type,
        }

        if have_enh_metrics:
            plume_mask_binary, enh_arr, uncert_arr, sens_arr = align_plume_to_enh_grid(
                l2b_tif, l2b_enh, l2b_uncert, l2b_sens
            )
            difficulty_metrics = compute_plume_metrics(
                enh_arr, uncert_arr, sens_arr, plume_mask_binary
            )
            
            training_cat = assign_training_category(difficulty_metrics, gas_type)
            difficulty_metrics["training_category"] = training_cat
            
            # Convert values safely for JSON serialization
            for k, v in difficulty_metrics.items():
                difficulty_metrics[k] = None if (np.isscalar(v) and pd.isna(v)) else (float(v) if not isinstance(v, (str, bool)) else v)
                
            plume_metadata = plume_metadata | plume_stats | difficulty_metrics
        else:
            plume_metadata["training_category"] = "uncategorized"
            plume_metadata = plume_metadata | plume_stats

        plume_metadata_list.append(plume_metadata)

    return plume_metadata_list


def save_hypercube(chip, path_without_ext, fmt="npy", dtype=np.float32):
    """
    Save a hypercube chip to disk in the specified format.
    
    Args:
        chip:             numpy array or xarray DataArray (H, W, C)
        path_without_ext: output path without file extension
        fmt:              "npy" for memory-mappable .npy, "hdf5" for single-chip HDF5
        dtype:            target dtype (default float32 to halve size vs float64)
    
    Returns:
        The path to the written file (with extension).
    """
    data = np.asarray(chip, dtype=dtype)

    if fmt == "npy":
        out_path = path_without_ext + ".npy"
        np.save(out_path, data)
        return out_path

    elif fmt == "hdf5":
        import h5py
        out_path = path_without_ext + ".h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset(
                "hypercube",
                data=data,
                chunks=data.shape,
                compression=None,
            )
        return out_path

    else:
        raise ValueError(f"Unknown format '{fmt}'. Supported: 'npy', 'hdf5'")


def load_hypercube(path, fmt=None):  
    """
    Load a hypercube chip from disk. Format is inferred from extension if not given.
    
    Args:
        path: file path (with extension)
        fmt:  "npy" or "hdf5". Inferred from extension if None.
    
    Returns:
        numpy array (H, W, C). For npy, returns a read-only memory-mapped view.
    """
    if fmt is None:
        if path.endswith(".npy"):
            fmt = "npy"
        elif path.endswith(".h5") or path.endswith(".hdf5"):
            fmt = "hdf5"
        else:
            raise ValueError(f"Cannot infer format from extension: {path}")

    if fmt == "npy":
        return np.load(path, mmap_mode="r")

    elif fmt == "hdf5":
        with h5py.File(path, "r") as f:
            return f["hypercube"][:]  # Load into memory before closing

    else:
        raise ValueError(f"Unknown format '{fmt}'. Supported: 'npy', 'hdf5'")


def save_instrument_metadata(dataset_dir, l1b_product):
    """
    Write instrument-level metadata (wavelengths, fwhm) once for the entire dataset.
    Skips writing if the file already exists.
    """
    out_path = os.path.join(dataset_dir, "instrument.json")
    if os.path.exists(out_path):
        return out_path

    instrument_metadata = {
        "wavelengths": np.asarray(l1b_product["wavelengths"]).tolist(),
        "fwhm": np.asarray(l1b_product["fwhm"]).tolist(),
        "num_bands": len(l1b_product["wavelengths"]),
    }
    os.makedirs(dataset_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(instrument_metadata, f)

    return out_path


def save_one_granule_to_dataset(granule_id, dataset_dir, return_chips = False, overwrite = False, 
                                cube_format="npy", chip_size=256, gas_type="ch4", run_mag1c=True):
    granule_dir = os.path.join(dataset_dir, f"{granule_id}_{gas_type}")
    granule_dir_if_mp = os.path.join(dataset_dir, f"{granule_id}_{gas_type}_multiple_plumes")
    run_mag1c = run_mag1c and (gas_type == "ch4")

    if overwrite == False and os.path.exists(granule_dir):
        print(f"Granule {granule_id} already exists in {dataset_dir}, skipping...")
        return
    if overwrite == False and os.path.exists(granule_dir_if_mp):
        print(f"Granule {granule_id} already exists in {dataset_dir}, skipping...")
        return
    if granule_id in KNOWN_BAD_GRANULE_IDS:
        print(f"Granule {granule_id} is known to be bad, skipping...")
        return
    
    temp_download_dir = os.path.join(dataset_dir, "_temp_download")
    l2b_jsons, l2b_tifs, l2b_enhs, l2b_uncerts, l2b_sens = download_one_l2b_set(
        granule_id, temp_dir=temp_download_dir, nasa_archive_dir=temp_download_dir, gas_type=gas_type
    )
        
    if len(l2b_jsons) == 0 or len(l2b_enhs) == 0:
        print(f"Granule {granule_id} has no plumes or enhancements, skipping...")
        shutil.rmtree(temp_download_dir, ignore_errors=True)
        return None
    
    if len(l2b_jsons) > 1:
        granule_dir = granule_dir_if_mp
    os.makedirs(granule_dir, exist_ok=True)
    
    nasa_dir = os.path.join(granule_dir, "nasa_plumes")
    shutil.move(temp_download_dir, nasa_dir)
    
    l1b_prod, mag1c_output = download_one_l1b_hypercube(granule_id, run_mag1c=run_mag1c) 
    save_instrument_metadata(dataset_dir, l1b_prod)
    
    granule_metadata = collate_granule_metadata(granule_id, l1b_prod, len(l2b_jsons), gas_type=gas_type)
    if len(l2b_enhs) > 0 and len(l2b_uncerts) > 0 and len(l2b_sens) > 0:
        plume_metadata_list = collate_plume_metadata(
            l2b_jsons,
            l2b_tifs,
            granule_id,
            gas_type=gas_type,
            l2b_enh=l2b_enhs[0],
            l2b_uncert=l2b_uncerts[0],
            l2b_sens=l2b_sens[0],
        )
    else:
        plume_metadata_list = collate_plume_metadata(
            l2b_jsons, l2b_tifs, granule_id, gas_type=gas_type
        )
    
    positive_chip_xy_list = []
    for i, plume_metadata in enumerate(plume_metadata_list):
        if plume_metadata.get("training_category") == "corrupted":
            print(f"Skipping chipping for granule {granule_id} plume {plume_metadata['plume_id']} due to corrupted EMIT arrays.")
            return None
        
        plume_mask_reprojected = project_plume_mask(l2b_tifs[i], l1b_prod)
        hypercube_chip, mag1c_chip, plume_mask_chip, l2b_enh_chip, (chip_plume_center_x, chip_plume_center_y), (chip_start_x, chip_start_y) = chip_plume(
            l1b_prod, plume_mask_reprojected, l2b_enhs[0], mag1c_layer=mag1c_output, chip_size=chip_size)
        positive_chip_xy_list.append((chip_start_x, chip_start_y))
        
        plume_metadata["chip_plume_center_x"] = int(chip_plume_center_x)
        plume_metadata["chip_plume_center_y"] = int(chip_plume_center_y)
        plume_metadata["chip_start_x"] = int(chip_start_x)
        plume_metadata["chip_start_y"] = int(chip_start_y)
        plume_metadata["chip_size"] = chip_size
        plume_metadata["gas_type"] = gas_type
        plume_difficulty = plume_metadata["training_category"]
        
        # print(plume_metadata["Plume ID"])
        # print(plume_difficulty)
        
        plume_dir = os.path.join(granule_dir, plume_metadata["plume_id"] + "_" + plume_difficulty)
        os.makedirs(plume_dir, exist_ok=True)
        with open(os.path.join(plume_dir, "plume_metadata.json"), "w") as f:
            json.dump(plume_metadata, f)
        
        #out_hypercube_path = os.path.join(plume_dir, "hypercube.npy")
        out_mag1c_path = os.path.join(plume_dir, "mag1c.npy")
        out_plume_mask_path = os.path.join(plume_dir, "plume_mask.npy")
        out_l2b_enh_path = os.path.join(plume_dir, "l2b_enh.npy")
        
        save_hypercube(hypercube_chip, os.path.join(plume_dir, "hypercube"), fmt=cube_format)
        if run_mag1c:
            np.save(out_mag1c_path, mag1c_chip)
        np.save(out_plume_mask_path, plume_mask_chip)
        np.save(out_l2b_enh_path, l2b_enh_chip)
    
        gc.collect()
    
    hypercube_chip_list, mag1c_chip_list, l2b_enh_chip_list, neg_chip_positions = chip_negatives(
        l1b_prod, l2b_enhs[0], mag1c_layer=mag1c_output, chips_to_generate=len(l2b_jsons), existing_positive_chips=positive_chip_xy_list, chip_size=chip_size)
    
    for i, (hypercube_chip, mag1c_chip, l2b_enh_chip) in enumerate(zip(hypercube_chip_list, mag1c_chip_list, l2b_enh_chip_list)):
        neg_dir = os.path.join(granule_dir, "negative_chip_" + str(i))
        os.makedirs(neg_dir, exist_ok=True)
        
        neg_metadata = {
            "chip_id": f"{granule_id}_neg_{i}",
            "granule_id": granule_id,
            "label": "negative",
            "chip_start_x": int(neg_chip_positions[i][0]),
            "chip_start_y": int(neg_chip_positions[i][1]),
            "chip_size": chip_size,
            "gas_type": gas_type,
        }
        with open(os.path.join(neg_dir, "chip_metadata.json"), "w") as f:
            json.dump(neg_metadata, f)

        save_hypercube(hypercube_chip, os.path.join(neg_dir, "hypercube"), fmt=cube_format)
        if run_mag1c:
            np.save(os.path.join(neg_dir, "mag1c.npy"), mag1c_chip)
        np.save(os.path.join(neg_dir, "l2b_enh.npy"), l2b_enh_chip)
    
    granule_metadata["num_positive_chips"] = len(positive_chip_xy_list)
    granule_metadata["num_negative_chips"] = len(hypercube_chip_list)
    with open(os.path.join(granule_dir, "granule_metadata.json"), "w") as f:
        json.dump(granule_metadata, f)
    
    # Build returns:
    _DIFFICULTY_KEYS = (
        "max_signal_to_uncertainty",
        "mean_plume_sensitivity",
        "plume_pixel_area",
        "background_std_dev",
    )
    index_rows = []
    for i, plume_metadata in enumerate(plume_metadata_list):
        plume_difficulty = plume_metadata["training_category"]
        dir_name = plume_metadata["plume_id"] + "_" + plume_difficulty
        row = {
            "id": f"{granule_id}_{plume_metadata['plume_id']}",
            "event_id": os.path.join(os.path.basename(granule_dir), dir_name),
            "granule_id": granule_id,
            "plume_id": plume_metadata["plume_id"],
            "label": "positive",
            "num_plume_pixels": plume_metadata["num_plume_pixels"],
            "training_category": plume_difficulty,
            "gas_type": gas_type,
        }
        for k in _DIFFICULTY_KEYS:
            row[k] = plume_metadata.get(k, np.nan)
        index_rows.append(row)

    for i in range(len(hypercube_chip_list)):
        dir_name = "negative_chip_" + str(i)
        row = {
            "id": f"{granule_id}_neg_{i}",
            "event_id": os.path.join(os.path.basename(granule_dir), dir_name),
            "granule_id": granule_id,
            "plume_id": None,
            "label": "negative",
            "num_plume_pixels": 0,
            "training_category": None,
            "gas_type": gas_type,
        }
        for k in _DIFFICULTY_KEYS:
            row[k] = np.nan
        index_rows.append(row)

    return pd.DataFrame(index_rows)


def build_dataset(granule_ids, dataset_dir, cube_format="npy", overwrite=False, gas_type="ch4", run_mag1c=True):
    """
    Build a complete dataset from a list of granule IDs.

    Downloads and chips each granule, archives NASA files, writes all metadata,
    and produces a dataset_index.csv for the training pipeline.

    Args:
        granule_ids:                list of granule ID strings (e.g. ['20230825T163454', ...])
        dataset_dir:                root directory for the dataset
        cube_format:                "npy" or "hdf5"
        overwrite:                  if False, skip granules whose directories already exist
        gas_type:                   "ch4" or "co2"
        run_mag1c:                  if True, run MAG1C on the L1B hypercube
    
    Returns:
        The full dataset index as a pandas DataFrame.
    """
    os.makedirs(dataset_dir, exist_ok=True)

    all_index_dfs = []
    for i, granule_id in enumerate(granule_ids):
        print(f"\n[{i+1}/{len(granule_ids)}] Processing {gas_type.upper()} granule {granule_id}...")
        try:
            result = save_one_granule_to_dataset(
                granule_id, dataset_dir, overwrite=overwrite, cube_format=cube_format, gas_type=gas_type, run_mag1c=run_mag1c
            )
            if result is not None and isinstance(result, pd.DataFrame):
                all_index_dfs.append(result)

        except Exception as e:
            print(f"  ERROR processing {granule_id}: {e}")
            continue

    if len(all_index_dfs) == 0:
        print("No granules were successfully processed.")
        return pd.DataFrame()

    dataset_index = pd.concat(all_index_dfs, ignore_index=True)
    csv_path = os.path.join(dataset_dir, "dataset_index.csv")
    dataset_index.to_csv(csv_path, index=False)
    print(f"\nDataset complete: {len(dataset_index)} chips across {len(all_index_dfs)} granules.")
    print(f"Index written to {csv_path}")

    return dataset_index