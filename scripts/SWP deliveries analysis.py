# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:57:42 2024

@author: armen
"""

import pandas as pd
excel_path = r"C:/Users/armen/Desktop/Drought Indicators - SoCal/Main/SWP annual total deliveries.xlsx"
df = pd.read_excel(excel_path, sheet_name='table b-5a', header=[0, 1, 2, 3, 4], index_col=0)
group_by_column = df.columns.get_level_values(3)
grouped_df = df.groupby(group_by_column, axis=1).sum()
#%%
# Filter columns where 'met' appears in the column names
filtered_columns = [col for col in df.columns if 'met' in col[3].lower()]
filtered_df = df[filtered_columns]
group_by_column = filtered_df.columns.get_level_values(1)
met_df = filtered_df.groupby(group_by_column, axis=1).sum()
met_df.plot()
met_df['Total'] = met_df.sum(axis=1)

#%%
import pandas as pd
excel_path = r"C:/Users/armen/Desktop/Drought Indicators - SoCal/Main/SWP annual total deliveries.xlsx"
table_b4 = pd.read_excel(excel_path, sheet_name='table b-4', header=[0, 1, 2], index_col=0)
met_table_a = table_b4

#%%

group_by_column = df.columns.get_level_values(1)
grouped_df = df.groupby(group_by_column, axis=1).sum()