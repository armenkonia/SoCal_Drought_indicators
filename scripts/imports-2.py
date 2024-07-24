# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:04:28 2024

@author: armen
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

years = [
    2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 
    2012, 2013, 2014, 2015, 2016, 2018, 2019, 2020]
swp_values = [
    1533.5, 1712.9, 1836.2, 1528.5, 1469.7, 1596.4, 1269.2, 985.7, 
    826.9, 900.7, 1170.4, 1060.8, 642.9, 456.4, 917.3, 1042.9, 921.5, 
    1039.6]
data = {
    'Year': years,
    'SWP': swp_values}
swp = pd.DataFrame(data)

#%%
sw_indicator = pd.read_csv('../../Indicators/surface_water_indicator.csv')
# socal_sw = sw_indicator[sw_indicator.HR_NAME == 'South Coast']
socal_sw = sw_indicator[sw_indicator.HR_NAME == 'Tulare Lake']
socal_sw.date = pd.to_datetime(socal_sw.date)
socal_sw['year'] = socal_sw['date'].dt.year
yearly_swdi_sw_TL = socal_sw.groupby('year')['SWDI'].mean().reset_index()
# yearly_swdi_sw.year = pd.to_datetime(yearly_swdi_sw.year)

socal_sw = sw_indicator[sw_indicator.HR_NAME == 'Sacramento River']
socal_sw.date = pd.to_datetime(socal_sw.date)
socal_sw['year'] = socal_sw['date'].dt.year
yearly_swdi_sw_SR = socal_sw.groupby('year')['SWDI'].mean().reset_index()

sw_indicator = pd.read_csv('../Indicators/surface_water_indicator.csv')
socal_sw = sw_indicator[sw_indicator.HR_NAME == 'South Coast']
socal_sw.date = pd.to_datetime(socal_sw.date)
socal_sw['year'] = socal_sw['date'].dt.year
yearly_swdi_sw_SC = socal_sw.groupby('year')['SWDI'].mean().reset_index()
#%%
imports_indicator = pd.read_csv('../Indicators/not preprocessed/total_storage_percentiles - imports.csv')
imports_indicator.date = pd.to_datetime(imports_indicator.date)
imports_indicator['year'] = imports_indicator['date'].dt.year
yearly_swdi_imports = imports_indicator.groupby('year')['SWDI'].mean().reset_index()
# yearly_swdi_imports.year = pd.to_datetime(yearly_swdi_imports.year)
#%%
data = pd.merge(swp,yearly_swdi_sw_SC,left_on='Year',right_on='year', suffixes=('_sw', '_sc'))
data = pd.merge(data,yearly_swdi_sw_SR,left_on='year',right_on='year', suffixes=('', '_SR'))
data = pd.merge(data,yearly_swdi_sw_TL,left_on='year',right_on='year', suffixes=('', '_TL'))

data = data.drop(columns='Year')
#%%
y = data.SWP
X = data[['SWDI', 'SWDI_SR', 'SWDI_TL']]#, 'exceptional_drought']]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

z = results.predict()

plt.plot(data.year, data.SWP, label = 'observed')
plt.plot(data.year, z, label = 'modeled')
# plt.ylim(0,5000000)
plt.legend()

comparison = pd.DataFrame([data.SWP,z]).transpose()