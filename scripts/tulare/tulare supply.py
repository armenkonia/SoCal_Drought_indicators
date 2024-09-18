# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:10:48 2024

@author: armen
"""

import pandas as pd
#Reading supply portfolio data (data is for water years)
supply_portfolios = pd.read_csv(r"C:\Users\armen\Downloads\supply_portfolios\supply_portfolios.csv")


tl_data = supply_portfolios.loc[supply_portfolios.HR=='Tulare Lake']
tl_data = tl_data.pivot(index = 'SupplyCategory', columns = 'Year', values = 'taf')
tl_data = tl_data.transpose().reset_index()
tl_data = tl_data.rename(columns={"Year": "wy"})

#Separating CVP data into local sources and imports
cvp_data = pd.read_csv(r"C:\Users\armen\Downloads\DWR_WaterBalances\cvp_annual_wy.csv")
cvp_data = cvp_data.rename(columns={'year': 'wy'})
cvp_data = cvp_data.rename(columns={'deliveries_wy_weighted': 'cvp_af'})
cvp_data['source'] = 'delta_imports'
cvp_data.loc[(cvp_data.cvp_branch=='Friant-Kern Canal') | (cvp_data.cvp_branch=='Madera Canal and Millerton Lake'), 'source'] = 'local_sj'
cvp_data.loc[(cvp_data.cvp_branch=='Sacramento River') | (cvp_data.cvp_branch=='Tehama-Colusa Canal'), 'source'] = 'local_sac'
cvp_data = cvp_data.groupby(['wy','source', 'hydrologic_region']).sum().reset_index()
tl_imports = cvp_data.loc[(cvp_data.source=='delta_imports')]
tl_imports = tl_imports.loc[tl_imports.hydrologic_region=='Tulare Lake']
tl_imports = tl_imports[['wy','cvp_af']]

tl_data = tl_data.merge(tl_imports, on = 'wy')
tl_data['cvp_af'] = 0.001*tl_data.cvp_af
tl_data['extra_federal'] = tl_data.Federal - tl_data.cvp_af
tl_data['final_local'] = tl_data.extra_federal + tl_data.LocalSupplies
tl_data['final_imports']=tl_data.cvp_af + tl_data.SWP
#%%
tl_local = cvp_data.loc[(cvp_data.source=='local_sj')]
tl_local = tl_local.loc[tl_local.hydrologic_region=='San Joaquin River']
tl_local = tl_local[['wy','cvp_af']]

sjv_local = sjv_local[['wy','cvp_af']].reset_index(drop=True)
sjv_imports = sjv_imports[['wy','cvp_af']].reset_index(drop=True)

sjv_federal = sjv_local.cvp_af + sjv_imports.cvp_af
sjv_federal = 0.001*sjv_federal

sjv_federal = cvp_data.loc[cvp_data.hydrologic_region=='San Joaquin River']
sjv_federal_summed = sjv_federal.groupby('wy').sum()
