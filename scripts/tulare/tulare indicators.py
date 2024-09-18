# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:50:19 2024

@author: armen
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

HR_region = 'Tulare Lake'
#%%
sw_indicator = pd.read_csv('../../Indicators/surface_water_indicator.csv')

socal_sw = sw_indicator[sw_indicator.HR_NAME == HR_region]
socal_sw['date'] = pd.to_datetime(socal_sw['date'])
socal_sw['year'] = socal_sw['date'].dt.year
socal_sw['month'] = socal_sw['date'].dt.month
socal_sw['water_year'] = socal_sw['year']
socal_sw.loc[socal_sw['month'] >= 10, 'water_year'] = socal_sw['year'] + 1 #this is to make it water year, if you want in calendar year cross this line
yearly_swdi_sw_sc = socal_sw.groupby('water_year')['SWDI'].mean().reset_index()
yearly_swdi_sw_sc = yearly_swdi_sw_sc[(yearly_swdi_sw_sc['water_year'] != 2017) & (yearly_swdi_sw_sc['water_year'] >= 2002) & (yearly_swdi_sw_sc['water_year'] <= 2020)]
# plt.plot(socal_sw['date'],socal_sw['SWDI'])
plt.plot(yearly_swdi_sw_sc['water_year'],yearly_swdi_sw_sc['SWDI'])

#%%
gw_indicator = pd.read_csv('../../Indicators/not preprocessed/state_wells_regional_analysis_seasonal.csv')
gw_indicator.date = pd.to_datetime(gw_indicator.date)
socal_gw = gw_indicator[gw_indicator.HR_NAME == HR_region]
socal_gw['year'] = socal_gw['date'].dt.year
socal_gw['month'] = socal_gw['date'].dt.month
socal_gw = socal_gw[['date','year','month','pctl_gwchange','pctl_cumgwchange','pctl_gwchange_corr','pctl_cumgwchange_corr','pctl_gwelev']]
# socal_gw.loc[socal_gw['month'] >= 10, 'water_year'] = socal_gw['year'] + 1

yearly_gw_sc = socal_gw.groupby('year').mean().reset_index()
yearly_gw_sc = yearly_gw_sc.dropna()

# socal_gw = socal_gw[['year','pctl_gwchange_corr','pctl_cumgwchange_corr']]
socal_gw = socal_gw[socal_gw.month == 9]
socal_gw = socal_gw[(socal_gw['year'] != 2017) & (socal_gw['year'] >= 2002) & (socal_gw['year'] <= 2020)]

#%%
sf_indicator = pd.read_csv('../../Indicators/streamflow_indicator.csv')
sf_col = sf_indicator.loc[sf_indicator.HR_NAME == HR_region]

sf_col.date = pd.to_datetime(sf_col.date)
sf_col['year'] = sf_col['date'].dt.year
sf_col['month'] = sf_col['date'].dt.month
sf_col['water_year'] = sf_col['year']
sf_col.loc[sf_col['month'] >= 10, 'water_year'] = sf_col['year'] + 1
yearly_sf_col = sf_col.groupby('water_year')['percentile'].mean().reset_index()
yearly_sf_col = yearly_sf_col[(yearly_sf_col['water_year'] != 2017) & (yearly_sf_col['water_year'] >= 2002) & (yearly_sf_col['water_year'] <= 2020)]

#%%