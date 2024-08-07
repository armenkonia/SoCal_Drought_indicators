# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:57:42 2024

@author: armen
"""

import pandas as pd
excel_path = r"C:/Users/armen/Desktop/Drought Indicators - SoCal/Main/SWP annual total deliveries.xlsx"
table_b5a = pd.read_excel(excel_path, sheet_name='table b-5a', header=[0, 1, 2, 3, 4], index_col=0)
table_b5a.columns = pd.MultiIndex.from_tuples([
    (a if 'Unnamed' not in a else '',
     b if 'Unnamed' not in b else '',
     c if 'Unnamed' not in c else '',
     d if 'Unnamed' not in d else '',
     e if 'Unnamed' not in e else '') for a, b, c, d, e in table_b5a.columns
])

table_b5a_no_total = pd.read_excel(excel_path, sheet_name='table b-5a modified', header=[0, 1, 2, 3, 4, 5], index_col=0)
table_b5a_no_total.columns = pd.MultiIndex.from_tuples([
    (a if 'Unnamed' not in a else '',
     b if 'Unnamed' not in b else '',
     c if 'Unnamed' not in c else '',
     d if 'Unnamed' not in d else '',
     e if 'Unnamed' not in e else '',
     f if 'Unnamed' not in f else '') for a, b, c, d, e, f in table_b5a_no_total.columns
])
table_b5a_no_total.columns = pd.MultiIndex.from_tuples([(a, b, c, f"{d} {e}".strip(), f) for a, b, c, d, e, f in table_b5a_no_total.columns])

#%%
# Filter columns where 'met' appears in the column names
filtered_columns = [col for col in table_b5a.columns if 'met' in col[3].lower()]
filtered_df = table_b5a[filtered_columns]
group_by_column = filtered_df.columns.get_level_values(1)
met_df = filtered_df.groupby(group_by_column, axis=1).sum()
met_df.plot()
met_df['Total'] = met_df.sum(axis=1)

table_b4 = pd.read_excel(excel_path, sheet_name='table b-4', header=[0, 1, 2], index_col=0)
met_table_a = table_b4

#%%
#branches
table_b5a_no_total_modified = table_b5a_no_total.copy()
table_b5a_no_total_modified.columns = pd.MultiIndex.from_tuples([(b, e) for a, b, c, d, e in table_b5a_no_total_modified.columns])
column_names = table_b5a_no_total_modified.columns
grouped_contractors_df = table_b5a_no_total_modified.groupby(column_names,axis=1).sum()
grouped_contractors_df.columns = pd.MultiIndex.from_tuples(grouped_contractors_df.columns)

#%%
#contractors
table_b5a_no_total_modified = table_b5a_no_total.copy()

contractor_column_names = table_b5a_no_total.columns.get_level_values(2)
# contractor_column_names = contractor_column_names.unique()
grouped_contractors_df = table_b5a_no_total.groupby(contractor_column_names, axis=1).sum()

#%%
#contractors and branch
table_b5a_no_total_modified = table_b5a_no_total.copy()
table_b5a_no_total_modified.columns = pd.MultiIndex.from_tuples([(b, d, e) for a, b, c, d, e in table_b5a_no_total_modified.columns])
column_names = table_b5a_no_total_modified.columns
grouped_contractors_df = table_b5a_no_total_modified.groupby(column_names,axis=1).sum()
grouped_contractors_df.columns = pd.MultiIndex.from_tuples(grouped_contractors_df.columns)

#%%
# branch and HR
table_b5a_no_total_modified = table_b5a_no_total.copy()
table_b5a_no_total_modified.columns = pd.MultiIndex.from_tuples([(b, e) for a, b, c, d, e in table_b5a_no_total_modified.columns])
column_names = table_b5a_no_total_modified.columns
grouped_contractors_df = table_b5a_no_total_modified.groupby(column_names,axis=1).sum()
grouped_contractors_df.columns = pd.MultiIndex.from_tuples(grouped_contractors_df.columns)
new_level_0 = grouped_contractors_df.columns.get_level_values(0).to_series().replace('', 'FEATHER RIVER AREA')
grouped_contractors_df.columns = pd.MultiIndex.from_arrays([new_level_0, grouped_contractors_df.columns.get_level_values(1)])
#%%

# Replace the empty column name with 'FEATHER RIVER AREA'
new_level_0 = grouped_contractors_df.columns.get_level_values(0).to_series().replace('', 'FEATHER RIVER AREA')
grouped_contractors_df.columns = pd.MultiIndex.from_arrays([new_level_0, grouped_contractors_df.columns.get_level_values(1)])

# HR to Branch and Regions mapping
hr_mapping = {
    'South Coast': ['WEST BRANCH', 'SANTA ANA DIVISION', 'MOJAVE DIVISION'],
    'Colorado River': ['SANTA ANA DIVISION', 'WEST BRANCH'],
    'South Lahontan': ['MOJAVE DIVISION', 'WEST BRANCH'],
    'Tulare Lake': ['COASTAL BRANCH', 'SOUTH SAN JOAQUIN DIVISION', 'TEHACHAPI DIVISION'],
    'Central Coast': ['COASTAL BRANCH'],
    'San Joaquin River': ['NORTH SAN JOAQUIN DIVISION'],
    'Sacramento River': ['FEATHER RIVER AREA'],
    'San Francisco Bay': ['DELTA DIVISION', 'NORTH SAN JOAQUIN DIVISION']
}

aggregated_df = pd.DataFrame(index=grouped_contractors_df.index)

# Loop through HR and their respective branches
for hr, branches in hr_mapping.items():
    columns_to_sum = pd.IndexSlice[branches, hr]
    aggregated_df[hr] = grouped_contractors_df.loc[:, columns_to_sum].sum(axis=1)
    
#%%
df = grouped_contractors_df.copy()
dfs_dict = {}

# Loop through HR and their respective branches to create separate DataFrames
for hr, branches in hr_mapping.items():
    columns_to_select = []
    for branch in branches:
        if (branch, hr) in df.columns:
            columns_to_select.append((branch, hr))
    if columns_to_select:
        dfs_dict[hr] = df[columns_to_select]
combined_df = pd.concat(dfs_dict, axis=1)
# combined_df.columns = pd.MultiIndex.from_tuples([(a, b) for a, b, c in combined_df.columns])
combined_df.columns = pd.MultiIndex.from_tuples([(b, c) for a, b, c in combined_df.columns])

#%%
original_columns = set(df.columns)

# Get the combined columns
combined_columns = set(combined_df.columns)

unused_columns = original_columns - combined_columns
unused_columns_df = df[list(unused_columns)]

# Sort columns based on the first level
sorted_columns = sorted(unused_columns_df.columns.get_level_values(0).unique())

# Create a new sorted DataFrame
sorted_df = unused_columns_df[sorted(sorted_columns, key=lambda x: unused_columns_df.columns.get_level_values(0).tolist().index(x))]
#%%
table_b5a_no_total_modified = table_b5a_no_total.copy()
table_b5a_no_total_modified_ssj = table_b5a_no_total_modified.loc[:, (slice(None), 'SOUTH SAN JOAQUIN DIVISION', slice(None), slice(None))]

