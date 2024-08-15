# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:51:58 2024

@author: armen
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../Other/watersuppliesinsocalfromdwrcoloradodiversions/supply portfolios only.csv')

df = df.iloc[:-1,:-3]
df.set_index('Years',inplace=True)
df = df[['Colorado','Groundwater', 'Imports', 'SWP']]
df['SWP+Imports'] = df['Imports'] + df['SWP']
df_corr = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()
#%%
# Plot the water sources over time
plt.figure(figsize=(14, 8))
for column in df.columns[1:]:
    plt.plot(df.index, df[column], label=column)

plt.title('Water Supply Sources Over Time')
plt.xlabel('Year')
plt.ylabel('Water Quantity')
plt.legend()
plt.show()
#%%
import statsmodels.api as sm

# Prepare the data for regression (e.g., predict 'Imports' based on other sources)
X = df[['Groundwater', 'Colorado', 'Desalination', 'Env', 'Federal', 'LocalSupplies', 'Other', 'Imports', 'TotalReturnFlowandReuse(TRFR)']]
y = df['SWP']

# Add a constant to the predictor variables
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

y_pred = model.predict(X)

# Display the summary of the regression model
print(model.summary())
# Step 5: Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(df.index, y, label='Actual Imports', marker='o')
plt.plot(df.index, y_pred, label='Predicted Imports', marker='x')
plt.xlabel('Year')
plt.ylabel('Imports')
plt.title('Actual vs Predicted Imports')
plt.legend()
plt.show()

#%%
import itertools
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Define the predictors and the target variable
predictors = ['Colorado', 'Desalination', 'Env', 'Federal', 'Groundwater', 'Imports',
              'LocalSupplies', 'Other', 'TotalReturnFlowandReuse(TRFR)']
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
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        
        # Calculate R-squared
        r2 = r2_score(y, y_pred)
        
        # Store the R-squared value in the dictionary
        r2_dict[tuple(combo)] = r2

#%%
# Find the combination with the highest R-squared value
best_combo = max(r2_dict, key=r2_dict.get)
best_r2 = r2_dict[best_combo]

# Refit the model with the best combination
X_best = df[list(best_combo)]
X_best = sm.add_constant(X_best)
best_model = sm.OLS(y, X_best).fit()
y_pred_best = best_model.predict(X_best)

# Plot the observed vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_best, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title(f'Observed vs. Predicted Values (Best R-squared: {best_r2:.4f})')
plt.show()

# Plot the predicted and observed values over the index
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Observed', color='b')
plt.plot(y.index, y_pred_best, label='Predicted', color='r', linestyle='--')
plt.xlabel('Index')
plt.ylabel('SWP')
plt.title(f'Observed and Predicted Values Over Time (Best R-squared: {best_r2:.4f})')
plt.legend()
plt.show()

#%%
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Define the predictors and the target variable
predictors = ['Colorado', 'Desalination', 'Env', 'Federal', 'Groundwater', 'Imports',
              'LocalSupplies', 'Other', 'TotalReturnFlowandReuse(TRFR)']
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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the predictors and the target variable
predictors = ['Colorado', 'Desalination', 'Env', 'Federal', 'Groundwater', 'Imports',
              'LocalSupplies', 'Other', 'TotalReturnFlowandReuse(TRFR)']
target = 'SWP'

# Prepare the target variable (y)
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[predictors], y, test_size=0.3, random_state=42)

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

