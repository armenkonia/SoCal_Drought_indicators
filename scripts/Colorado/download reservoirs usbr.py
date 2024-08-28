# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:49:53 2024

@author: armen
"""
import pandas as pd
import matplotlib.pyplot as plt

#17 is storage, 42 is total release
#923 is lake havasu
#921 is lake mead
#922 is lake mohave
df_mead = pd.read_csv('https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/17.csv')
df_powell = pd.read_csv('https://www.usbr.gov/uc/water/hydrodata/reservoir_data/919/csv/17.csv')
df_havasu = pd.read_csv('https://www.usbr.gov/uc/water/hydrodata/reservoir_data/923/csv/17.csv')
df_mohave = pd.read_csv('https://www.usbr.gov/uc/water/hydrodata/reservoir_data/922/csv/17.csv')
df_mead.datetime = pd.to_datetime(df_mead.datetime)
df_powell.datetime = pd.to_datetime(df_powell.datetime)
df_havasu.datetime = pd.to_datetime(df_havasu.datetime)
df_mohave.datetime = pd.to_datetime(df_mohave.datetime)

#%%

# Create a 3x2 grid of subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6),sharey=True)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

# Plot df_mead on the first row, spanning both columns
axes[0, 0].plot(df_mead['datetime'], df_mead['storage'])  # Replace 'value_column' with the actual data column
axes[0, 0].set_title('Lake Mead')
# axes[0, 0].set_ylabel('Value')
# axes[0, 1].axis('off')  # Hide the second subplot on this row

# Plot df_powell on the second row, spanning both columns
axes[0, 1].plot(df_powell['datetime'], df_powell['storage'])  # Replace 'value_column' with the actual data column
axes[0, 1].set_title('Lake Powell')
# axes[0, 1].set_ylabel('Value')
# axes[0, 1].axis('off')  # Hide the second subplot on this row

# Plot df_havasu on the third row, first column
axes[1, 0].plot(df_havasu['datetime'], df_havasu['storage'])  # Replace 'value_column' with the actual data column
axes[1, 0].set_title('Lake Havasu')
# axes[1, 0].set_ylabel('Value')

# Plot df_mohave on the third row, second column
axes[1, 1].plot(df_mohave['datetime'], df_mohave['storage'])  # Replace 'value_column' with the actual data column
axes[1, 1].set_title('Lake Mohave')
# axes[1, 1].set_ylabel('Value')

# Adjust layout
plt.tight_layout()
plt.show()

#%%
df_mead.set_index('datetime', inplace=True)
df_powell.set_index('datetime', inplace=True)
df_havasu.set_index('datetime', inplace=True)
df_mohave.set_index('datetime', inplace=True)

df_mead_m = df_mead.resample('YE').sum()
df_powell_m = df_powell.resample('YE').sum()
df_havasu_m = df_havasu.resample('YE').sum()
df_mohave_m = df_mohave.resample('YE').sum()

# Filter data to only include dates from 1980 onward
df_mead_m = df_mead_m[(df_mead_m.index >= '2007-01-01') & (df_mead_m.index < '2024-08-01')]
df_powell_m = df_powell_m[(df_powell_m.index >= '2007-01-01') & (df_powell_m.index < '2024-08-01')]
df_havasu_m = df_havasu_m[(df_havasu_m.index >= '2007-01-01') & (df_havasu_m.index < '2024-08-01')]
df_mohave_m = df_mohave_m[(df_mohave_m.index >= '2007-01-01') & (df_mohave_m.index < '2024-08-01')]

#%%
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6),sharey=True)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

# Plot df_mead on the first row, spanning both columns
axes[0, 0].plot(df_mead_m.index, df_mead_m['storage'])  # Replace 'value_column' with the actual data column
axes[0, 0].set_title('Lake Mead')
# axes[0, 0].set_ylabel('Value')
# axes[0, 1].axis('off')  # Hide the second subplot on this row

# Plot df_powell on the second row, spanning both columns
axes[0, 1].plot(df_powell_m.index, df_powell_m['storage'])  # Replace 'value_column' with the actual data column
axes[0, 1].set_title('Lake Powell')
# axes[0, 1].set_ylabel('Value')
# axes[0, 1].axis('off')  # Hide the second subplot on this row

# Plot df_havasu on the third row, first column
axes[1, 0].plot(df_havasu_m.index, df_havasu_m['storage'])  # Replace 'value_column' with the actual data column
axes[1, 0].set_title('Lake Havasu')
# axes[1, 0].set_ylabel('Value')

# Plot df_mohave on the third row, second column
axes[1, 1].plot(df_mohave_m.index, df_mohave_m['storage'])  # Replace 'value_column' with the actual data column
axes[1, 1].set_title('Lake Mohave')
# axes[1, 1].set_ylabel('Value')

# Adjust layout
plt.tight_layout()
plt.show()

#%%
df_mead['percent full'] = df_mead['storage']/max(df_mead['storage'])
df_powell['percent full'] = df_powell['storage']/max(df_powell['storage'])
df_havasu['percent full'] = df_havasu['storage']/df_havasu['storage'].max(skipna=True)
df_mohave['percent full'] = df_mohave['storage']/max(df_mohave['storage'])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6),sharey=True)

axes[0, 0].plot(df_mead['datetime'], df_mead['percent full'])  # Replace 'value_column' with the actual data column
axes[0, 0].set_title('Lake Mead')
axes[0, 1].plot(df_powell['datetime'], df_powell['percent full'])  # Replace 'value_column' with the actual data column
axes[0, 1].set_title('Lake Powell')
axes[1, 0].plot(df_havasu['datetime'], df_havasu['percent full'])  # Replace 'value_column' with the actual data column
axes[1, 0].set_title('Lake Havasu')
axes[1, 1].plot(df_mohave['datetime'], df_mohave['percent full'])  # Replace 'value_column' with the actual data column
axes[1, 1].set_title('Lake Mohave')

plt.tight_layout()
plt.show()