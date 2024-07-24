# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:28:33 2024

@author: armen
"""

import pandas as pd

swp_allocations = pd.read_csv('../allocations/SWP Allocations.csv')
imports_indicator = pd.read_csv('../../Indicators/not preprocessed/total_storage_percentiles - imports.csv')

swp_allocations['DATE'] = pd.to_datetime(swp_allocations['DATE'])
swp_allocations['MONTH'] = swp_allocations.DATE.dt.month

imports_indicator['date'] = pd.to_datetime(imports_indicator['date'])
imports_indicator['YEAR'] = imports_indicator.date.dt.year
imports_indicator['MONTH'] = imports_indicator.date.dt.month

df = pd.merge(swp_allocations, imports_indicator, on=['YEAR','MONTH'])

imports_indicator['SWDI'].plot()
swp_allocations['A'].plot()

#%%
df = pd.merge(swp_allocations, imports_indicator, on=['YEAR','MONTH'], how='outer')
df = df[[ 'date','YEAR','month','ALLOCATION','SWDI_y']]
df = df.sort_values(by='date')

df_filled = df.fillna(method='ffill')
