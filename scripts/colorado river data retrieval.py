# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:34:19 2024

@author: armen
"""

import dataretrieval.nwis as nwis
site = '09429000'

df = nwis.get_record(sites=site, service='dv', start='2000-01-01', end='2020-12-01')
df3 = nwis.get_record(sites=site, service='stat')

#%%
from dataretrieval import nwis
from IPython.display import display

# Retrieve the statistics
palo_verde = nwis.get_stats(sites=["09429000"], parameterCd=["00060"], statReportType="monthly")[0]
df = nwis.get_record(sites=site, service='site')

all_american = nwis.get_stats(sites=["09527700"], parameterCd=["00060"], statReportType="monthly")[0]

coachella = nwis.get_stats(sites=["09527590"], parameterCd=["00060"], statReportType="monthly")[0]

#%%
import requests
import pandas as pd
from io import StringIO

# URL to retrieve data from
url = 'https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi=2068&tstp=MN&t1=1982-01-01&t2=2021-02-01&table=R&mrid=0&format=1'

# Send HTTP GET request
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Extract <PRE> formatted data
    data_start = response.text.find('BEGIN DATA') + len('BEGIN DATA\n')
    data_end = response.text.find('</PRE>')
    data_text = response.text[data_start:data_end].strip()

    # Use StringIO to simulate a file object
    # Read CSV-like data into a DataFrame
    df = pd.read_csv(StringIO(data_text), skipinitialspace=True)

    # Display or process the DataFrame as needed
    print(df.head())  # Display the first few rows of the DataFrame
    
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
