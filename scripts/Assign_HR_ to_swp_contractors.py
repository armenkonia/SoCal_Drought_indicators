# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 08:54:10 2024

@author: armen
"""

import geopandas as gpd
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
shape_path_filename = r"..\Dataset\i03_WaterDistricts\i03_WaterDistricts.shp"
ca_wd = gpd.read_file(shape_path_filename).to_crs("EPSG:4326")


swp_contractors = pd.DataFrame(data = {
    "AGENCYNAME": [
        "Antelope Valley - East Kern Water Agency", "Solano County Water Agency",
        "Napa County Flood Control and Water Conservation District", "Empire West Side Irrigation District",
        "Kings County", "Mojave Water Agency", "Santa Barbara County Flood Control and Water Conservation District",
        "Ventura County Watershed Protection District", "Alameda County Water District SOI",
        "Dudley Ridge Water District", "Desert Water Agency", "Plumas County Flood Control and Water Conservation District",
        "Oak Flat Water District", "Kern County Water Agency", "Santa Clara Valley Water District",
        "San Luis Obispo County Flood Control And Water Conservation District", "Littlerock Creek Irrigation District",
        "San Gorgonio Pass Water Agency", "Crestline - Lake Arrowhead Water Agency", "Palmdale Water District",
        "Metropolitan Water District Of Southern California", "San Bernardino Valley Municipal Water District",
        "Santa Clarita Valley Water Agency", "Yuba City", "Tulare Lake Basin Water Storage District",
        "Alameda County Flood Control District  Zone 7", "Butte County of", "Coachella Valley Water District",
        "San Gabriel Valley Municipal Water District"]})

swp_wd = ca_wd[ca_wd['AGENCYNAME'].isin(swp_contractors['AGENCYNAME'])]

shape_path_filename = r"..\Dataset\HRs\i03_Hydrologic_Regions.shp"
hrs = gpd.read_file(shape_path_filename).to_crs("EPSG:4326")

shape_path_filename = r"..\Dataset\i17_StateWaterProject_Centerline\i17_StateWaterProject_Centerline.shp"
swp_centerline = gpd.read_file(shape_path_filename).to_crs("EPSG:4326")

swp_wd = swp_wd.to_crs("EPSG:3857")
hrs = hrs.to_crs("EPSG:3857")
swp_centerline = swp_centerline.to_crs("EPSG:3857")

#%%
# this is to figure out which HR each contractor is in
swp_wd_with_hrs = gpd.sjoin(swp_wd, hrs, how="inner", predicate='within')
swp_wd['centroid'] = swp_wd.geometry.centroid
swp_wd_centroids = gpd.GeoDataFrame(swp_wd[['AGENCYNAME']], geometry=swp_wd['centroid'], crs=swp_wd.crs)
swp_wd_with_hrs = gpd.sjoin(swp_wd_centroids, hrs, how="inner", predicate='within')

#%%
fig, ax = plt.subplots(figsize=(12, 8))
hrs.plot(ax=ax, cmap='gray', alpha=0.2, edgecolor='none')
hrs.boundary.plot(ax=ax, facecolor='none', edgecolor='black')

swp_wd.plot(ax=ax, column='AGENCYNAME', cmap='tab20', edgecolor='black', alpha=1)

# Plot the centroids
swp_wd_centroids.plot(ax=ax, color='red', markersize=50, label='Centroids')

# Customize plot
plt.title('Map of SWP Contractors and HRS with Centroids')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

# Annotate centroids with AGENCYNAME
texts = []
for idx, row in swp_wd_centroids.iterrows():
    text = ax.annotate(
        row['AGENCYNAME'],
        xy=(row['geometry'].x, row['geometry'].y),
        xytext=(3, 3),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white'),
        arrowprops=dict(arrowstyle="->", color='black')
    )
    texts.append(text)

# Adjust text labels to prevent overlap
adjust_text(texts, ax=ax, force_text=0.8, expand_text=1.1, expand_points=1.2)

plt.show()
#%%
# Create a mapping from AGENCYNAME to numbers
agency_name_to_number = {name: idx + 1 for idx, name in enumerate(swp_wd['AGENCYNAME'].unique())}
number_to_agency_name = {v: k for k, v in agency_name_to_number.items()}

# Create a column with numbers
swp_wd_centroids['number'] = swp_wd_centroids['AGENCYNAME'].map(agency_name_to_number)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
hrs.plot(ax=ax, cmap='gray', alpha=0.1, edgecolor='none')
hrs.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
swp_wd.plot(ax=ax, column='AGENCYNAME', cmap='tab20', edgecolor='black', alpha=1)
swp_centerline.plot(ax=ax, edgecolor='red', linewidth=4)

# Annotate centroids with numbers
for idx, row in swp_wd_centroids.iterrows():
    ax.annotate(
        row['number'],
        xy=(row['geometry'].x, row['geometry'].y),
        xytext=(3, 3),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white'),
        arrowprops=dict(arrowstyle="->", color='black')
    )

# Create a legend for the numbers
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=1, linestyle='None') for _ in agency_name_to_number]
labels = [f'{num}: {agency}' for num, agency in number_to_agency_name.items()]
ax.legend(handles, labels, title='AGENCYNAME', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.title('Map of SWP Contractors')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

plt.show()
#%%
# joined = gpd.sjoin(swp_wd, hrs, how='inner', predicate='intersects')
# joined['intersection_area'] = joined.geometry.area
# swp_wd['total_area'] = swp_wd.geometry.area
# joined = joined.merge(swp_wd[['AGENCYNAME', 'total_area']], on='AGENCYNAME')
# joined['area_percentage'] = (joined['intersection_area'] / joined['total_area']) * 100
# majority_hr = joined.loc[joined.groupby('AGENCYNAME')['area_percentage'].idxmax()]


# #%%
# # Plot the results
# fig, ax = plt.subplots(figsize=(12, 8))
# hrs.plot(ax=ax, cmap='tab20', alpha=0.2, edgecolor='black')
# swp_wd.plot(ax=ax, column='AGENCYNAME', cmap='tab20', edgecolor='black', alpha=1)

# # Annotate each SWP contractor with the majority hydrologic region
# texts = []
# for idx, row in swp_wd.iterrows():
#     # Get the centroid of the geometry for annotation
#     centroid = row['geometry'].centroid
#     text = ax.annotate(
#         row['AGENCYNAME'],
#         xy=(centroid.x, centroid.y),
#         xytext=(1, 1),  # Further reduce the offset to bring labels even closer
#         textcoords='offset points',
#         fontsize=6,
#         bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white'),
#         arrowprops=dict(arrowstyle="->", color='black')
#     )
#     texts.append(text)


# # Adjust text labels to prevent overlap
# from adjustText import adjust_text
# adjust_text(texts, ax=ax, force_text=0.8, expand_text=1.1, expand_points=1.2)

# # Customize plot
# plt.title('Map of SWP Contractors and HRS with Majority Hydrologic Regions')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.tight_layout()

# plt.show()

#%%
import matplotlib.lines as mlines

# Create a mapping from HR_NAME to numbers
hr_name_to_number = {name: idx + 1 for idx, name in enumerate(hrs['HR_NAME'].unique())}
number_to_hr_name = {v: k for k, v in hr_name_to_number.items()}

# Add a column with numbers to the `hrs` GeoDataFrame (if needed)
hrs['number'] = hrs['HR_NAME'].map(hr_name_to_number)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
hrs.plot(ax=ax, cmap='tab10', alpha=0.2, edgecolor='none')
hrs.boundary.plot(ax=ax, facecolor='none', edgecolor='black')

# Annotate with numbers
for idx, row in hrs.iterrows():
    ax.annotate(
        row['number'],
        xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
        xytext=(3, 3),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white'),
        arrowprops=dict(arrowstyle="->", color='black')
    )

# Create a legend for the numbers
handles = [mlines.Line2D([], [], color='gray', marker='o', markersize=10, linestyle='None') for _ in hr_name_to_number]
labels = [f'{num}: {name}' for num, name in number_to_hr_name.items()]
ax.legend(handles, labels, title='HR_NAME', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.title('Map with HR_NAME and Annotations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

plt.show()
