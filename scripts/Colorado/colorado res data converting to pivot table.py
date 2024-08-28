# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:54:35 2024

@author: armen
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


col_res_indicator = pd.read_csv('individual_reservoir_percentiles.csv')

#%%
# Assuming `col_res_indicator` is your DataFrame
stations = col_res_indicator['station'].unique()

# Create a dictionary to store the list for each unique station
station_dict = {}

for station_name in stations:
    # Filter the DataFrame for each station
    selected_res = col_res_indicator.loc[col_res_indicator['station'] == station_name]
    
    # Add the filtered DataFrame or specific column values to the dictionary
    station_dict[station_name] = selected_res  # or selected_res['percentile'].tolist() for just the list of percentiles

#%%
merged_df = pd.DataFrame()

for station_name, station_df in station_dict.items():
    # Select only the 'date' and 'percentile' columns, rename 'percentile' to include the station name
    station_df = station_df[['date', 'percentile']].rename(columns={'percentile': f'percentile_{station_name}'})
    
    if merged_df.empty:
        # Initialize merged_df with the first DataFrame
        merged_df = station_df
    else:
        # Merge the DataFrames on 'date'
        merged_df = pd.merge(merged_df, station_df, on='date', how='outer')
        
merged_df.plot()