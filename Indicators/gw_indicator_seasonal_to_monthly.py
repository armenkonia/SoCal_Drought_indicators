# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:46:54 2024

@author: armen
"""

import pandas as pd
import matplotlib.pyplot as plt
gw = pd.read_csv('not preprocessed/state_wells_regional_analysis.csv')
gw['date'] = pd.to_datetime(gw['date'])
gw.set_index('date', inplace=True)

# List of columns to interpolate
columns_to_interpolate = [
    'wlm_gse', 'gwe', 'gse_gwe', 'gse', 'rpe', 'well_depth', 'gwchange',
    'pctl_gwchange', 'half_gwchange', 'cumgwchange', 'pctl_cumgwchange',
    'pctl_gwelev', 'pctl_gwchange_corr','pctl_cumgwchange_corr'
]

gw_monthly = pd.DataFrame()
for hr_name in gw['HR_NAME'].unique():
    df_hr = gw[gw['HR_NAME'] == hr_name]    
    df_resampled = df_hr.resample('M').asfreq()
    df_resampled_inferred = df_resampled.infer_objects(copy=False)
    df_interpolated = df_resampled_inferred[columns_to_interpolate].interpolate(method='linear')
    df_filled = df_resampled_inferred.ffill()
    df_combined = df_filled.copy()
    df_combined[columns_to_interpolate] = df_interpolated[columns_to_interpolate]
    df_combined.reset_index(inplace=True)
    gw_monthly = pd.concat([gw_monthly, df_combined])
    
gw_monthly.to_csv('state_wells_regional_analysis_monthly.csv')
gw_monthly = gw_monthly[['date',	'HR_NAME','pctl_gwchange','pctl_gwchange_corr','pctl_gwelev','pctl_cumgwchange','pctl_cumgwchange_corr']]
gw_monthly.to_csv('groundwater_indicator_monthly.csv')
