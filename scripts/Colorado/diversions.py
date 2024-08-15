# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:10:43 2024

@author: armen
"""

import requests
import pandas as pd
from io import StringIO
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
sdi = '2991'
url = f"https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi={sdi}&tstp=MN&t1=1980-10-01T00:00&t2=2024-08-12T00:00&table=R&mrid=0&format=csv"
response = requests.get(url)
raw_data = response.text
clean_data = raw_data.replace('<BR>', '\n').replace('<HTML>', '').replace('<HEAD>', '').replace('<TITLE>Bureau of Reclamation HDB Data</TITLE>', '').replace('</HEAD>', '').replace('<BODY>', '').replace('<PRE>', '').replace('</BODY>', '').replace('</HTML>', '').replace('</PRE>', '')
data = StringIO(clean_data.strip())
df_iid = pd.read_csv(data)
df_iid['DATETIME'] = pd.to_datetime(df_iid['DATETIME'], format='%m/%d/%Y %H:%M')

