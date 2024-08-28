# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:10:43 2024

@author: armen
"""

import requests
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

url = "https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi=2068&tstp=MN&t1=1980-10-01T00:00&t2=2024-08-12T00:00&table=R&mrid=0&format=csv"
response = requests.get(url)
raw_data = response.text
clean_data = raw_data.replace('<BR>', '\n').replace('<HTML>', '').replace('<HEAD>', '').replace('<TITLE>Bureau of Reclamation HDB Data</TITLE>', '').replace('</HEAD>', '').replace('<BODY>', '').replace('<PRE>', '').replace('</BODY>', '').replace('</HTML>', '').replace('</PRE>', '')
data = StringIO(clean_data.strip())
df = pd.read_csv(data)
df['     SDI_2068'] = df['     SDI_2068'].astype(float)
df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%m/%d/%Y %H:%M')

def get_water_year(date):
    return date.year if date.month < 10 else date.year + 1
df['Water_Year'] = df['DATETIME'].apply(get_water_year)
df_wy = df.groupby('Water_Year')['     SDI_2068'].sum().reset_index()

df['Calendar_Year'] = df['DATETIME'].dt.year
df_cy = df.groupby('Calendar_Year')['     SDI_2068'].sum().reset_index()
#%%
sdi = '20675'
url = f"https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi={sdi}&tstp=MN&t1=2000-10-01T00:00&t2=2024-08-12T00:00&table=R&mrid=0&format=csv"
response = requests.get(url)
raw_data = response.text
clean_data = raw_data.replace('<BR>', '\n').replace('<HTML>', '').replace('<HEAD>', '').replace('<TITLE>Bureau of Reclamation HDB Data</TITLE>', '').replace('</HEAD>', '').replace('<BODY>', '').replace('<PRE>', '').replace('</BODY>', '').replace('</HTML>', '').replace('</PRE>', '')
data = StringIO(clean_data.strip())
df_sd = pd.read_csv(data)
df_sd['DATETIME'] = pd.to_datetime(df_sd['DATETIME'], format='%m/%d/%Y %H:%M')

#%%
sdi = '2771'
url = f"https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi={sdi}&tstp=MN&t1=1980-10-01T00:00&t2=2024-08-12T00:00&table=R&mrid=0&format=csv"
response = requests.get(url)
raw_data = response.text
clean_data = raw_data.replace('<BR>', '\n').replace('<HTML>', '').replace('<HEAD>', '').replace('<TITLE>Bureau of Reclamation HDB Data</TITLE>', '').replace('</HEAD>', '').replace('<BODY>', '').replace('<PRE>', '').replace('</BODY>', '').replace('</HTML>', '').replace('</PRE>', '')
data = StringIO(clean_data.strip())
df_mwd = pd.read_csv(data)
df_mwd['DATETIME'] = pd.to_datetime(df_mwd['DATETIME'], format='%m/%d/%Y %H:%M')

#%%
# mwd, pvid, iid, cvwd
diversion_sdi = ['2771','2980','2991','2994']
con_use_sdi = ['2972','2982','2993','2996']

#%%
import requests
import pandas as pd
from io import StringIO

diversion_sdi = ['2771', '2980', '2991', '2994']
con_use_sdi = ['2972', '2982', '2993', '2996']

all_sdi = {
    'diversion': diversion_sdi,
    'consumptive_use': con_use_sdi
}

def fetch_data(sdi, start_date, end_date, file_format='csv'):
    url = f"https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi={sdi}&tstp=MN&t1={start_date}T00:00&t2={end_date}T00:00&table=R&mrid=0&format={file_format}"
    response = requests.get(url)
    
    if file_format == 'csv':
        raw_data = response.text
        clean_data = raw_data.replace('<BR>', '\n').replace('<HTML>', '').replace('<HEAD>', '').replace('<TITLE>Bureau of Reclamation HDB Data</TITLE>', '').replace('</HEAD>', '').replace('<BODY>', '').replace('<PRE>', '').replace('</BODY>', '').replace('</HTML>', '').replace('</PRE>', '')
        data = StringIO(clean_data.strip())
        df = pd.read_csv(data)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%m/%d/%Y %H:%M')
    
    elif file_format == 'json':
        raw_data = response.json()
        df = pd.json_normalize(raw_data)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%m/%d/%Y %H:%M')
    
    else:
        raise ValueError("Unsupported format. Please choose 'csv' or 'json'.")
    
    return df

start_date = '1900-10-01'
end_date = '2024-08-12'
# df = fetch_data(sdi, start_date, end_date, file_format='csv')

data_dict = {}

for data_type, sdi_list in all_sdi.items():
    data_dict[data_type] = {}
    for sdi in sdi_list:
        data_dict[data_type][sdi] = fetch_data(sdi, start_date, end_date, 'csv')
#%%
data_types = list(data_dict.keys())
diversions_dict = data_dict[data_types[0]]
con_use_dict = data_dict[data_types[1]]

merged_df = None
for df in diversions_dict.values():
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on='DATETIME', how='outer')

merged_df = merged_df.set_index('DATETIME')
# merged_df.columns = ['MWD','PVID','IID','CVWD']
merged_df = merged_df.apply(pd.to_numeric, errors='coerce')
# merged_df.plot()

merged_df_yr = merged_df.groupby(pd.Grouper(freq='YE')).sum()
merged_df_yr = merged_df_yr.iloc[:-1,:]       
# merged_df_yr.plot()
#%%
filtered_df_y = merged_df_yr.loc[merged_df_yr.index >= '1990']

filtered_df_y.plot()
plt.title('Yearly Data from 1990 to Present')
plt.xlabel('Year')
plt.ylabel('Values')
plt.show()
#%%
filtered_df = merged_df.loc[merged_df.index >= '1990']
filtered_df = filtered_df.reset_index()
filtered_df['DATETIME'] = pd.to_datetime(filtered_df['DATETIME'])
filtered_df['year'] = filtered_df['DATETIME'].dt.year
filtered_df['month'] = filtered_df['DATETIME'].dt.month
filtered_df['water_year'] = filtered_df['year']
filtered_df.loc[filtered_df['month'] >= 10, 'water_year'] = filtered_df['year'] + 1 #this is to make it water year, if you want in calendar year cross this line
filtered_df_wy = filtered_df.groupby('water_year')[['MWD', 'PVID', 'IID', 'CVWD']].sum().reset_index()
# plt.plot(socal_sw['date'],socal_sw['SWDI'])
plt.plot(filtered_df_wy['water_year'],filtered_df_wy['MWD'])
#%%
data_types = list(data_dict.keys())
diversions_dict = data_dict[data_types[0]]
con_use_dict = data_dict[data_types[1]]

merged_df = None
for df in con_use_dict.values():
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on='DATETIME', how='outer')

merged_df = merged_df.set_index('DATETIME')
merged_df.columns = ['MWD','PVID','IID','CVWD']
merged_df = merged_df.apply(pd.to_numeric, errors='coerce')
# merged_df.plot()

merged_df_yr_cu = merged_df.groupby(pd.Grouper(freq='YE')).sum()
merged_df_yr_cu = merged_df_yr.iloc[:-1,:]       

#%%
merged_df_yr_cu['MWD'].plot()
merged_df_yr['MWD'].plot()
