# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:32:04 2024

@author: armen
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('../allocations/Historical SWP allocations 1996-2024 042324.csv')
df = df[['DATE', 'ALLOCATION']]
df['DATE'] = pd.to_datetime(df['DATE'], format='mixed')
df['ALLOCATION'] = df['ALLOCATION'].str.rstrip('%').astype(float)
df = df.sort_values(by='DATE')
df['YEAR'] = df['DATE'].dt.year
plt.plot(df['DATE'], df['ALLOCATION'])
plt.title('Allocation Over Time')
plt.xlabel('Date')
plt.ylabel('Allocation (%)')
df = df[['YEAR','DATE', 'ALLOCATION']]
# df.to_csv('SWP Allocations.csv')
#%%
# Extract month and adjust year for water year
df['MONTH'] = df['DATE'].dt.month
df['WATER_YEAR'] = df['DATE'].apply(lambda x: x.year if x.month not in [11, 12] else x.year + 1)

# Filter out rows where 'WATER_MONTH' is 8 or greater
annual_allocations = df[df['MONTH'] < 9]
annual_allocations = annual_allocations.loc[annual_allocations.groupby('YEAR')['DATE'].idxmax()]
annual_allocations.reset_index(drop=True, inplace=True)
#%%
df.set_index('DATE', inplace=True)
full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
full_df = pd.DataFrame(index=full_date_range)
merged_df = full_df.merge(df, left_index=True, right_index=True, how='left')
merged_df['ALLOCATION'] = merged_df['ALLOCATION'].ffill()
merged_df['YEAR'] = merged_df.index.year
merged_df = merged_df.reset_index().rename(columns={'index': 'DATE'})

plt.plot(merged_df['DATE'], merged_df['ALLOCATION'])
plt.title('Allocation Over Time')
plt.xlabel('Date')
plt.ylabel('Allocation (%)')
merged_df.set_index('DATE', inplace=True)

#%%

annual_features = merged_df.resample('A').agg(
    allocation_sum=('ALLOCATION', 'sum'),
    allocation_mean=('ALLOCATION', 'mean'),
    allocation_max=('ALLOCATION', 'max'),
    allocation_min=('ALLOCATION', 'min'),
    allocation_std=('ALLOCATION', 'std'),
    allocation_median=('ALLOCATION', 'median'),
    allocation_25th=('ALLOCATION', lambda x: np.percentile(x, 25)),
    allocation_75th=('ALLOCATION', lambda x: np.percentile(x, 75)),
    zero_allocation_days=('ALLOCATION', lambda x: (x == 0).sum())
)
annual_features['year'] = annual_features.index.year
annual_average = merged_df.resample('A').median()
#%%
imports_indicator = pd.read_csv('../../Indicators/not preprocessed/total_storage_percentiles - imports.csv')
imports_indicator.date = pd.to_datetime(imports_indicator.date)
imports_indicator['year'] = imports_indicator['date'].dt.year
yearly_swdi_imports = imports_indicator.groupby('year')['SWDI'].mean().reset_index()
# yearly_swdi_imports = imports_indicator.groupby('year')['res_percentile'].mean().reset_index()

#%%
imports_allocation_based = pd.merge(yearly_swdi_imports,annual_features,left_on='year',right_on='year', suffixes=('_sw', '_imports'))
imports_allocation_based.set_index('year',inplace=True)
for col in imports_allocation_based.columns[1:]:
    imports_allocation_based[col] = imports_allocation_based[col] * imports_allocation_based['SWDI']

#%%
# Data as lists
years = [
    2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 
    2012, 2013, 2014, 2015, 2016, 2018, 2019, 2020]
swp_values = [
    1533.5, 1712.9, 1836.2, 1528.5, 1469.7, 1596.4, 1269.2, 985.7, 
    826.9, 900.7, 1170.4, 1060.8, 642.9, 456.4, 917.3, 1042.9, 921.5, 
    1039.6]
data = {
    'year': years,
    'SWP': swp_values}
swp = pd.DataFrame(data)

#%%
data = pd.merge(swp,imports_allocation_based,left_on='year',right_on='year', suffixes=('_sw', '_imports'))
data = data.set_index('year')

y = data.SWP
X = data.iloc[:,1:]/100
# X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

z = results.predict()

bar_width = 0.4
index = np.arange(len(data))

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(index - bar_width / 2, data.SWP, bar_width, label='Observed')
plt.bar(index + bar_width / 2, z, bar_width, label='Modeled')
plt.xticks(index, data.index, rotation=45)
plt.legend()

comparison = pd.DataFrame([y,z]).transpose()

#%%
df = data
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Define the predictors and the target variable
predictors = [ 'SWDI', 'allocation_mean', 'allocation_std', 'allocation_median',
       'zero_allocation_days']

target = 'SWP'

# Prepare the target variable (y)
y = df[target]

# Dictionary to store R-squared values for each combination
r2_dict = {}

# Iterate through all combinations of the predictors
for i in range(1, len(predictors) + 1):
    for combo in itertools.combinations(predictors, i):
        combo = list(combo)
        X = df[combo]
        X = sm.add_constant(X)
        
        # Linear Regression
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        r2_dict[('Linear', tuple(combo))] = r2

        # Polynomial Regression (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(df[combo])
        poly_model = LinearRegression().fit(X_poly, y)
        y_pred_poly = poly_model.predict(X_poly)
        r2_poly = r2_score(y, y_pred_poly)
        r2_dict[('Polynomial', tuple(combo))] = r2_poly

        # Exponential Regression
        X_exp = sm.add_constant(np.log1p(df[combo]))  # log1p to handle zero values
        exp_model = sm.OLS(np.log1p(y), X_exp).fit()
        y_pred_exp = np.expm1(exp_model.predict(X_exp))
        r2_exp = r2_score(y, y_pred_exp)
        r2_dict[('Exponential', tuple(combo))] = r2_exp

        # Logarithmic Regression
        X_log = sm.add_constant(np.log1p(df[combo]))  # log1p to handle zero values
        log_model = sm.OLS(y, X_log).fit()
        y_pred_log = log_model.predict(X_log)
        r2_log = r2_score(y, y_pred_log)
        r2_dict[('Logarithmic', tuple(combo))] = r2_log

# Find the model and combination with the highest R-squared value
best_model, best_combo = max(r2_dict, key=r2_dict.get)
best_r2 = r2_dict[(best_model, best_combo)]

# Refit the best model with the best combination
X_best = df[list(best_combo)]
if best_model == 'Linear':
    X_best = sm.add_constant(X_best)
    best_model_fit = sm.OLS(y, X_best).fit()
    y_pred_best = best_model_fit.predict(X_best)
# elif best_model == 'Polynomial':
#     poly = PolynomialFeatures(degree=2)
#     X_best_poly = poly.fit_transform(X_best)
#     best_model_fit = LinearRegression().fit(X_best_poly, y)
#     y_pred_best = best_model_fit.predict(X_best_poly)
elif best_model == 'Exponential':
    X_best_exp = sm.add_constant(np.log1p(X_best))
    best_model_fit = sm.OLS(np.log1p(y), X_best_exp).fit()
    y_pred_best = np.expm1(best_model_fit.predict(X_best_exp))
elif best_model == 'Logarithmic':
    X_best_log = sm.add_constant(np.log1p(X_best))
    best_model_fit = sm.OLS(y, X_best_log).fit()
    y_pred_best = best_model_fit.predict(X_best_log)

# Plot the observed vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_best, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title(f'Observed vs. Predicted Values ({best_model} Regression, Best R-squared: {best_r2:.4f})')
plt.show()

# Plot the predicted and observed values over the index
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Observed', color='b')
plt.plot(y.index, y_pred_best, label='Predicted', color='r', linestyle='--')
plt.xlabel('Index')
plt.ylabel('SWP')
plt.title(f'Observed and Predicted Values Over Time ({best_model} Regression, Best R-squared: {best_r2:.4f})')
plt.legend()
plt.show()
#%%
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the predictors and the target variable
predictors = [ 'SWDI', 'allocation_mean', 'allocation_std', 'allocation_median',
       'zero_allocation_days']
target = 'SWP'

# Prepare the target variable (y)
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[predictors], y, test_size=0.7, shuffle=False)

# Dictionary to store R-squared values for each combination
r2_dict = {}

# Iterate through all combinations of the predictors
for i in range(1, len(predictors) + 1):
    for combo in itertools.combinations(predictors, i):
        combo = list(combo)
        X_train_combo = X_train[combo]
        X_test_combo = X_test[combo]
        
        # Linear Regression
        X_train_linear = sm.add_constant(X_train_combo)
        X_test_linear = sm.add_constant(X_test_combo)
        model = sm.OLS(y_train, X_train_linear).fit()
        y_pred = model.predict(X_test_linear)
        r2 = r2_score(y_test, y_pred)
        r2_dict[('Linear', tuple(combo))] = r2

        # Polynomial Regression (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train_combo)
        X_test_poly = poly.transform(X_test_combo)
        poly_model = LinearRegression().fit(X_train_poly, y_train)
        y_pred_poly = poly_model.predict(X_test_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        r2_dict[('Polynomial', tuple(combo))] = r2_poly

        # Exponential Regression
        X_train_exp = sm.add_constant(np.log1p(X_train_combo))  # log1p to handle zero values
        X_test_exp = sm.add_constant(np.log1p(X_test_combo))
        exp_model = sm.OLS(np.log1p(y_train), X_train_exp).fit()
        y_pred_exp = np.expm1(exp_model.predict(X_test_exp))
        r2_exp = r2_score(y_test, y_pred_exp)
        r2_dict[('Exponential', tuple(combo))] = r2_exp

        # Logarithmic Regression
        X_train_log = sm.add_constant(np.log1p(X_train_combo))  # log1p to handle zero values
        X_test_log = sm.add_constant(np.log1p(X_test_combo))
        log_model = sm.OLS(y_train, X_train_log).fit()
        y_pred_log = log_model.predict(X_test_log)
        r2_log = r2_score(y_test, y_pred_log)
        r2_dict[('Logarithmic', tuple(combo))] = r2_log

# Find the model and combination with the highest R-squared value
best_model, best_combo = max(r2_dict, key=r2_dict.get)
best_r2 = r2_dict[(best_model, best_combo)]

# Refit the best model with the best combination on the entire dataset
X_best = df[list(best_combo)]
if best_model == 'Linear':
    X_best = sm.add_constant(X_best)
    best_model_fit = sm.OLS(y, X_best).fit()
    y_pred_best = best_model_fit.predict(X_best)
elif best_model == 'Polynomial':
    poly = PolynomialFeatures(degree=2)
    X_best_poly = poly.fit_transform(X_best)
    best_model_fit = LinearRegression().fit(X_best_poly, y)
    y_pred_best = best_model_fit.predict(X_best_poly)
elif best_model == 'Exponential':
    X_best_exp = sm.add_constant(np.log1p(X_best))
    best_model_fit = sm.OLS(np.log1p(y), X_best_exp).fit()
    y_pred_best = np.expm1(best_model_fit.predict(X_best_exp))
elif best_model == 'Logarithmic':
    X_best_log = sm.add_constant(np.log1p(X_best))
    best_model_fit = sm.OLS(y, X_best_log).fit()
    y_pred_best = best_model_fit.predict(X_best_log)

# Plot the observed vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_best, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title(f'Observed vs. Predicted Values ({best_model} Regression, Best R-squared: {best_r2:.4f})')
plt.show()

# Plot the predicted and observed values over the index
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Observed', color='b')
plt.plot(y.index, y_pred_best, label='Predicted', color='r', linestyle='--')
plt.xlabel('Index')
plt.ylabel('SWP')
plt.title(f'Observed and Predicted Values Over Time ({best_model} Regression, Best R-squared: {best_r2:.4f})')
plt.legend()
plt.show()

#%%
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the predictors and the target variable
predictors = [ 'SWDI', 'allocation_mean', 'allocation_std', 'allocation_median',
       'zero_allocation_days']
target = 'SWP'

# Prepare the target variable (y)
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[predictors], y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(df[predictors], y, test_size=0.2, shuffle=False)

# Dictionary to store R-squared and MAPE values for each combination
results_dict = {}

# Iterate through all combinations of the predictors
for i in range(1, len(predictors) + 1):
    for combo in itertools.combinations(predictors, i):
        combo = list(combo)
        X_train_combo = X_train[combo]
        X_test_combo = X_test[combo]
        
        # Linear Regression
        X_train_linear = sm.add_constant(X_train_combo)
        X_test_linear = sm.add_constant(X_test_combo)
        model = sm.OLS(y_train, X_train_linear).fit()
        y_pred = model.predict(X_test_linear)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        results_dict[('Linear', tuple(combo))] = (r2, mape)

        # Polynomial Regression (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train_combo)
        X_test_poly = poly.transform(X_test_combo)
        poly_model = LinearRegression().fit(X_train_poly, y_train)
        y_pred_poly = poly_model.predict(X_test_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        mape_poly = mean_absolute_percentage_error(y_test, y_pred_poly)
        results_dict[('Polynomial', tuple(combo))] = (r2_poly, mape_poly)

        # Exponential Regression
        X_train_exp = sm.add_constant(np.log1p(X_train_combo))  # log1p to handle zero values
        X_test_exp = sm.add_constant(np.log1p(X_test_combo))
        exp_model = sm.OLS(np.log1p(y_train), X_train_exp).fit()
        y_pred_exp = np.expm1(exp_model.predict(X_test_exp))
        r2_exp = r2_score(y_test, y_pred_exp)
        mape_exp = mean_absolute_percentage_error(y_test, y_pred_exp)
        results_dict[('Exponential', tuple(combo))] = (r2_exp, mape_exp)

        # Logarithmic Regression
        X_train_log = sm.add_constant(np.log1p(X_train_combo))  # log1p to handle zero values
        X_test_log = sm.add_constant(np.log1p(X_test_combo))
        log_model = sm.OLS(y_train, X_train_log).fit()
        y_pred_log = log_model.predict(X_test_log)
        r2_log = r2_score(y_test, y_pred_log)
        mape_log = mean_absolute_percentage_error(y_test, y_pred_log)
        results_dict[('Logarithmic', tuple(combo))] = (r2_log, mape_log)

# Find the model and combination with the highest R-squared value
best_model_r2, best_combo_r2 = max(results_dict, key=lambda x: results_dict[x][0])
best_r2 = results_dict[(best_model_r2, best_combo_r2)][0]

# Find the model and combination with the lowest MAPE value
best_model_mape, best_combo_mape = min(results_dict, key=lambda x: results_dict[x][1])
best_mape = results_dict[(best_model_mape, best_combo_mape)][1]

# Refit the best models with the best combinations on the entire dataset
def refit_and_predict(best_model, best_combo):
    X_best = df[list(best_combo)]
    if best_model == 'Linear':
        X_best = sm.add_constant(X_best)
        best_model_fit = sm.OLS(y, X_best).fit()
        y_pred_best = best_model_fit.predict(X_best)
    elif best_model == 'Polynomial':
        poly = PolynomialFeatures(degree=2)
        X_best_poly = poly.fit_transform(X_best)
        best_model_fit = LinearRegression().fit(X_best_poly, y)
        y_pred_best = best_model_fit.predict(X_best_poly)
    elif best_model == 'Exponential':
        X_best_exp = sm.add_constant(np.log1p(X_best))
        best_model_fit = sm.OLS(np.log1p(y), X_best_exp).fit()
        y_pred_best = np.expm1(best_model_fit.predict(X_best_exp))
    elif best_model == 'Logarithmic':
        X_best_log = sm.add_constant(np.log1p(X_best))
        best_model_fit = sm.OLS(y, X_best_log).fit()
        y_pred_best = best_model_fit.predict(X_best_log)
    return y_pred_best

y_pred_best_r2 = refit_and_predict(best_model_r2, best_combo_r2)
y_pred_best_mape = refit_and_predict(best_model_mape, best_combo_mape)

# Plot the observed vs. predicted values for the best R-squared model
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_best_r2, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title(f'Observed vs. Predicted Values ({best_model_r2} Regression, Best R-squared: {best_r2:.4f})')
plt.show()

# Plot the predicted and observed values over the index for the best R-squared model
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Observed', color='b')
plt.plot(y.index, y_pred_best_r2, label='Predicted', color='r', linestyle='--')
plt.xlabel('Index')
plt.ylabel('SWP')
plt.title(f'Observed and Predicted Values Over Time ({best_model_r2} Regression, Best R-squared: {best_r2:.4f})')
plt.legend()
plt.show()

# Plot the observed vs. predicted values for the best MAPE model
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_best_mape, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title(f'Observed vs. Predicted Values ({best_model_mape} Regression, Best MAPE: {best_mape:.4f})')
plt.show()

# Plot the predicted and observed values over the index for the best MAPE model
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Observed', color='b')
plt.plot(y.index, y_pred_best_mape, label='Predicted', color='r', linestyle='--')
plt.xlabel('Index')
plt.ylabel('SWP')
plt.title(f'Observed and Predicted Values Over Time ({best_model_mape} Regression, Best MAPE: {best_mape:.4f})')
plt.legend()
plt.show()
