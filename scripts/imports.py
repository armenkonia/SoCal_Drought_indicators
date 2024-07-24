# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:13:05 2024

@author: armen
"""


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

sw_indicator = pd.read_csv('../Indicators/surface_water_indicator.csv')
# socal_sw = sw_indicator[sw_indicator.HR_NAME == 'South Coast']
socal_sw = sw_indicator[sw_indicator.HR_NAME == 'Tulare Lake']
socal_sw.date = pd.to_datetime(socal_sw.date)

# Data as lists
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
socal_sw['year'] = socal_sw['date'].dt.year
yearly_swdi_sw = socal_sw.groupby('year')['SWDI'].mean().reset_index()
# yearly_swdi_sw.year = pd.to_datetime(yearly_swdi_sw.year)

#%%
imports_indicator = pd.read_csv('../Indicators/not preprocessed/total_storage_percentiles - imports.csv')
imports_indicator.date = pd.to_datetime(imports_indicator.date)
imports_indicator['year'] = imports_indicator['date'].dt.year
yearly_swdi_imports = imports_indicator.groupby('year')['SWDI'].mean().reset_index()
# yearly_swdi_imports.year = pd.to_datetime(yearly_swdi_imports.year)
#%%
data = pd.merge(swp,yearly_swdi_sw,left_on='Year',right_on='year')
data = pd.merge(data,yearly_swdi_imports,left_on='year',right_on='year', suffixes=('_sw', '_imports'))
data = data.drop(columns='Year')
#%%

#%%
y = data.SWP
X = data[['SWDI_sw', 'SWDI_imports']]#, 'exceptional_drought']]
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