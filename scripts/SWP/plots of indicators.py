# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:20:11 2024

@author: armen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_excel(r"C:\Users\armen\Desktop\Drought Indicators - SoCal\Supplies\Colorado\Colorado USBR Analysis.xlsx", sheet_name='Sheet1', header=1)
df = df.iloc[:33, :8]
df_1991_2002 = df.iloc[:12, :8]
df_2002_2023 = df.iloc[12:33, :8]

# df = df_1991_2002

variables = ['SWDI delta', 'SWDI SC', 'pctl_gwchange_corr', 'SWDI Colorado']
titles = ['South Coast vs SWDI delta', 'South Coast vs SWDI SC', 
          'South Coast vs gw change', 'South Coast vs SWDI Colorado']

y_min = df['South Coast (MWD)'].min()
y_max = df['South Coast (MWD)'].max()

fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex= True, sharey= True)
axs = axs.ravel()  # Flatten the 2D array of axes

i = 0
for var in variables:
    df_selected = df[['South Coast (MWD)', var]].dropna()
    x = df_selected[var].values.reshape(-1, 1)
    y = df_selected['South Coast (MWD)'].values

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)

    axs[i].scatter(x, y, color='blue')
    axs[i].plot(x, y_pred, color='red', label=f'R² = {r2:.2f}')
    axs[i].set_xlabel(var)
    axs[i].set_ylabel('South Coast (MWD)')
    axs[i].set_title(titles[i])
    axs[i].legend(loc='upper right')
    axs[i].set_ylim([y_min*1.05, y_max*1.05])
    i += 1

plt.suptitle('1991-2023', fontsize=16)
plt.tight_layout()
plt.show()
