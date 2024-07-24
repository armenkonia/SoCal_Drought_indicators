# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:49:13 2024

@author: armen
"""


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
gw_indicator = pd.read_csv('../../Indicators/not preprocessed/state_wells_regional_analysis.csv')
gw_indicator.date = pd.to_datetime(gw_indicator.date)
gw_indicator['Year'] = gw_indicator['date'].dt.year
socal_gw = gw_indicator[gw_indicator.HR_NAME == 'South Coast']
socal_gw = socal_gw[['Year','pctl_gwelev','pctl_gwchange','pctl_cumgwchange','pctl_gwchange_corr','pctl_cumgwchange_corr']]
socal_gw = socal_gw.dropna()

for col in socal_gw.columns:
    if col != 'Year':
        socal_gw[col] = pd.to_numeric(socal_gw[col], errors='coerce')

yearly_gwi = socal_gw.groupby('Year').mean().reset_index()
#%%

# Data as lists
years = [
    2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 
    2012, 2013, 2014, 2015, 2016, 2018, 2019, 2020]
gw_values = [
    1897.6, 1542.7, 1476.3, 1237.6, 1739.9, 1802.4, 1697.1, 1744.5, 
    1408.2, 1351, 1484.1, 1824.2, 1986.1, 1462.3, 1331, 1579.3, 
    1414.9, 1362.1
]
data = {
    'Year': years,
    'GW': gw_values}
gw = pd.DataFrame(data)
#%%
data = pd.merge(gw,yearly_gwi,left_on='Year',right_on='Year', suffixes=('_sw', '_imports'))
data = data.set_index('Year')

y = data.GW
X = data[['pctl_gwchange_corr', 'pctl_cumgwchange_corr']]
# X = data[['pctl_gwchange_corr']]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

z = results.predict()

bar_width = 0.4
index = np.arange(len(data))

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(index - bar_width / 2, data['GW'], bar_width, label='Observed')
plt.bar(index + bar_width / 2, z, bar_width, label='Modeled')
plt.xticks(index, data.index, rotation=45)
plt.legend()

comparison = pd.DataFrame([y,z]).transpose()
#%%
# gw_indicator = pd.read_csv('../Indicators/not preprocessed/state_wells_regional_analysis.csv')
# gw_indicator['date'] = pd.to_datetime(gw_indicator['date'])
# gw_indicator.set_index('date', inplace=True)
# socal_gw = gw_indicator[gw_indicator.HR_NAME == 'South Coast']
# socal_gw = socal_gw.resample('M').interpolate(method='linear')
# socal_gw = socal_gw.reset_index()
# socal_gw['Year'] = socal_gw['date'].dt.year
# socal_gw = socal_gw[['Year','pctl_gwelev','pctl_gwchange','pctl_cumgwchange','pctl_gwchange_corr','pctl_cumgwchange_corr']]
# socal_gw = socal_gw.dropna()
# yearly_gwi = socal_gw.groupby('Year').mean().reset_index()

