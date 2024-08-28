#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:49:49 2023

@author: alvar
"""
"""
Description:
    This script calculates imports indicators using reservoir and snow data for the delta region. 
        The script performs the following tasks:
        1. Calculates monthly individual reservoir percentile values using a 1 month analysis period.
        2. Calculates monthly individual snow gauge percentile values using a 1 month analysis period.
        3. Calculated monthly reservoir percentile values at the hydrologic region scale using a 1 month analysis period.
        4. Calculates monthly total storage (sum of reservoir storage and snow) percentile at the hydrologic region scale using a 1 month analysis period.
"""
import pandas as pd
from percentile_average_function import func_for_tperiod
import os
import matplotlib.pyplot as plt

#reading reservoir and snow data
reservoir_data = pd.read_csv('reservoirs.csv')

#converting date to datetime
reservoir_data['date'] = pd.to_datetime(reservoir_data.date)


#Subseting data for delta_exporting_basins
reservoir_data['value'] = pd.to_numeric(reservoir_data['value'], errors='coerce')


#First we obtain the percentiles for individual reservoirs
res_ind = func_for_tperiod(reservoir_data, date_column = 'date', value_column = 'value',
                          input_timestep = 'M', analysis_period = '1M',function = 'percentile',
                          grouping_column='station', correcting_no_reporting = True,
                          correcting_column = 'capacity',baseline_start_year = 1991, 
                          baseline_end_year = 2020)

#Then we obtain the aggregated values at the hydrologic region scale for storage
res_hr = func_for_tperiod(reservoir_data, date_column = 'date', value_column = 'value',
                          input_timestep = 'M', analysis_period = '1M',function = 'percentile',
                          grouping_column='HR_NAME', correcting_no_reporting = True,
                          correcting_column = 'capacity',baseline_start_year = 1991, 
                          baseline_end_year = 2020)

#Correcting date after obtaining aggregated data per hydrologic region
res_hr['month'] = res_hr['date'].dt.month
res_hr['year'] = res_hr['date'].dt.year
res_hr['date'] = pd.to_datetime(dict(year=res_hr.year, month=res_hr.month, day=1))

res_hr['percentile'].plot()

res_ind.to_csv('individual_reservoir_percentiles.csv')
res_hr.to_csv('total_storage_percentiles.csv')

#%%
stations = res_ind.station.unique()  
# selected_stations = [stations[i] for i in [1, 3]]
selected_stations = [stations[i] for i in [0, 2]]

fig, ax = plt.subplots()
for station_name in selected_stations:
    selected_res = res_ind[res_ind['station'] == station_name]
    ax.plot(selected_res['date'], selected_res['percentile'], label=station_name)

# ax.plot(res_hr['date'], res_hr['percentile'], label='res_hr', color='black', linestyle='--')

ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Percentile')
ax.set_title('Percentile by Station')

fig.autofmt_xdate()