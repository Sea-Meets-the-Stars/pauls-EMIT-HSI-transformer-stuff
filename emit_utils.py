import earthaccess
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from skimage import feature, color
from matplotlib import colors as mcolors

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
        count=max_count
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
    
