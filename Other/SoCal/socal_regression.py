#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:29:00 2024

@author: alvar
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv('socal_water_data.csv')
data['popeff'] = data.population * data.efficiency



y = data.water_use
X = data[['population', 'precipitation', 'efficiency', 'sb2020', 'drought']]#, 'exceptional_drought']]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

z = results.predict()

plt.plot(data.year, data.water_use, label = 'observed')
plt.plot(data.year, z, label = 'modeled')
plt.ylim(0,5000000)
plt.legend()

comparison = pd.DataFrame([data.water_use,z]).transpose()
