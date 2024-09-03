# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:30:28 2024

@author: armen
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('../Other/allocations/Historical SWP allocations 1996-2024 042324.csv')
df = df[['DATE', 'ALLOCATION']]
df['DATE'] = pd.to_datetime(df['DATE'], format='mixed')
df['ALLOCATION'] = df['ALLOCATION'].str.rstrip('%').astype(float)
df = df.sort_values(by='DATE')
df['Year'] = df['DATE'].dt.year
plt.plot(df['DATE'], df['ALLOCATION'])
plt.title('Allocation Over Time')
plt.xlabel('Date')
plt.ylabel('Allocation (%)')
df = df[['Year','DATE', 'ALLOCATION']]
# df.to_csv('SWP Allocations.csv')
df['Month'] = df['DATE'].dt.month
#%%
# Extract month and adjust year for water year
df['WATER_YEAR'] = df['DATE'].apply(lambda x: x.year if x.month not in [11, 12] else x.year + 1)

# Filter out rows where 'WATER_MONTH' is 8 or greater
annual_allocations = df[df['Month'] < 9]
annual_allocations = annual_allocations.loc[annual_allocations.groupby('Year')['DATE'].idxmax()]
annual_allocations.reset_index(drop=True, inplace=True)
#%%
# imports_indicator = pd.read_csv('C:/Users/armen/Desktop/Drought Impacts/Indicators/not preprocessed/total_storage_percentiles - imports.csv')
imports_indicator = pd.read_csv(r"C:\Users\armen\Desktop\Drought Impacts - Final\Indicators\not preprocessed\total_storage_percentiles - imports.csv")
imports_indicator.date = pd.to_datetime(imports_indicator.date)
imports_indicator = imports_indicator[['reservoir_storage','date','export_basin', 'SWDI']]
imports_indicator['Year'] = imports_indicator['date'].dt.year
imports_indicator['Month'] = imports_indicator['date'].dt.month

yearly_swdi_imports = imports_indicator.groupby('Year')['SWDI'].mean().reset_index()

#%%
df = df.merge(imports_indicator, on=['Year','Month'],how='inner')
df = df[['Year','Month', 'ALLOCATION','SWDI','reservoir_storage']]

#%%
annual_allocations = annual_allocations.merge(imports_indicator, on=['Year','Month'],how='inner')
annual_allocations = annual_allocations[['Year','Month', 'ALLOCATION','SWDI','reservoir_storage']]