# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:06:17 2024

@author: armen
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from tqdm import tqdm

excel_path = r"C:\Users\armen\Desktop\Drought Indicators - SoCal\Analysis\Indicator Analysis Results.xlsx"
df = pd.read_excel(excel_path, sheet_name='Main', header=[1], index_col=0)

df_log = df.copy()
for col in df_log.columns:
    df_log[col] = np.log10(df_log[col] + 1e-8)
df_log = df_log.rename(columns=lambda x: f'log_{x}')

df_exp = df.copy()
for col in df_exp.columns:
    df_exp[col] = np.power(10, df_exp[col])
df_exp = df_exp.rename(columns=lambda x: f'exp_{x}')

all_predictors = ['SWDI SC', 'SWDI delta imports', 'SWDI colorado', 'gw elevation indicator', 'gw pumping intensity']

merged_df = pd.merge(df, df_exp.iloc[:,5:], left_index = True, right_index = True)
merged_df = pd.merge(merged_df, df_log.iloc[:,5:], left_index = True, right_index = True)

#%%
all_predictors = merged_df.columns[5:]

all_combinations = []
max_variables = 3  # Limit the combinations to 3 variables
for r in range(1, min(len(all_predictors), max_variables) + 1):
    for combo in combinations(all_predictors, r):
        all_combinations.append(combo)
 
# This is for all combinations - no limit on the number of combinations
# all_combinations = [] 
# for r in range(1, len(all_predictors) + 1):
#     for combo in combinations(all_predictors, r):
#         all_combinations.append(combo)

# Combining actual with log and exp
df = merged_df.copy()  
df['Category'] = df.index.to_series().apply(lambda x: 1 if 2002 <= x <= 2007 else 0)
      
results = [] 
for combo in tqdm(all_combinations):
    X = df[list(combo)]
    # X = df[list(combo) + ['Category']]
    y = df['Groundwater']
    # y = np.log10(y + 1e-8)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    r_squared = model.rsquared
    adjusted_r_squared = model.rsquared_adj
    
    # Store results
    results.append({
        'Predictors': combo,
        'R-squared': r_squared,
        'Adjusted R-squared': adjusted_r_squared,
        'Number of Variables': len(combo)
    })

results_df = pd.DataFrame(results)
predictors_df = results_df['Predictors'].apply(pd.Series)
predictors_df.columns = [f'Predictor_{i+1}' for i in predictors_df.columns]
df_separated = pd.concat([results_df.drop(columns='Predictors'), predictors_df], axis=1)

df_separated.to_csv('combination results.csv')
#%%

X = df[['SWDI SC']]
y = df['SWP']
# y = df['SWP']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
r_squared = model.rsquared
