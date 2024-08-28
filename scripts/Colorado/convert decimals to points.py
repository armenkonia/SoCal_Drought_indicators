# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:56:08 2024

@author: armen
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
df = pd.read_csv(r"C:\Users\armen\Desktop\Drought Indicators - SoCal\Main\Colorado\Datasets\USBR River Stations.csv")
df_usbr = df.loc[df['site_metadata.description'] == 'USBR Site - telemetry']

# def convert_ddm_to_dd(ddm):
#     """Convert DDM (Degrees and Decimal Minutes) to Decimal Degrees."""
#     ddm = ddm.replace("'", "")  # Remove the trailing apostrophe
#     if " " in ddm:
#         degrees, minutes = ddm.split(" ")
#         return float(degrees) + float(minutes) / 60
#     return float(ddm)  # Return as float if it's already in decimal degrees

# # Apply conversion
# df_usbr.loc[:, 'site_metadata.lat'] = df_usbr['site_metadata.lat'].apply(convert_ddm_to_dd)
# df_usbr.loc[:, 'site_metadata.longi'] = df_usbr['site_metadata.longi'].apply(lambda x: convert_ddm_to_dd(x))

# df.update(df_usbr[['site_metadata.lat', 'site_metadata.longi']])

#%%
def convert_to_decimal_degrees(coord):
    """Convert coordinates from DDM, DMS to Decimal Degrees, or do nothing if already in Decimal Degrees."""
    
    def dms_to_dd(degrees, minutes, seconds):
        """Convert DMS to Decimal Degrees."""
        return float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    
    # Check if coord is already in Decimal Degrees format
    if '.' in coord:
        try:
            # Attempt to convert to float to validate it's a proper decimal degree
            return float(coord)
        except ValueError:
            # If conversion fails, proceed with other formats
            pass
    
    coord = coord.replace("'", "")  # Remove the trailing apostrophe

    if " " in coord and len(coord.split(" ")) == 3:
        # Assuming DMS format: "degrees minutes seconds"
        parts = coord.split(" ")
        degrees = parts[0]
        minutes = parts[1]
        seconds = parts[2]
        return dms_to_dd(degrees, minutes, seconds)
    elif " " in coord:
        # Assuming DDM format: "degrees minutes"
        degrees, minutes = coord.split(" ")
        return float(degrees) + float(minutes) / 60
    else:
        # If the input doesn't match any known format, return it as is
        return coord

# Test examples
ddm_lat = "33 13.256'"
dms_lat = "36 51 53"
dd_lat = "33.220933"
invalid_coord = "invalid"

print(f"DDM to Decimal Degrees: {convert_to_decimal_degrees(ddm_lat)}")
print(f"DMS to Decimal Degrees: {convert_to_decimal_degrees(dms_lat)}")
print(f"Decimal Degrees: {convert_to_decimal_degrees(dd_lat)}")
print(f"Invalid Coordinate: {convert_to_decimal_degrees(invalid_coord)}")
#%%
# Apply the function to both latitude and longitude columns
df['site_metadata.lat'] = df['site_metadata.lat'].apply(convert_to_decimal_degrees)
df['site_metadata.longi'] = df['site_metadata.longi'].apply(convert_to_decimal_degrees)

# # Apply conversion
# df_usbr.loc[:, 'site_metadata.lat'] = df_usbr['site_metadata.lat'].apply(convert_to_decimal_degrees)
# df_usbr.loc[:, 'site_metadata.longi'] = df_usbr['site_metadata.longi'].apply(lambda x: convert_to_decimal_degrees(x))
df.to_csv(r"C:\Users\armen\Desktop\Drought Indicators - SoCal\Main\Colorado\Datasets\USBR River Stations1.csv")
