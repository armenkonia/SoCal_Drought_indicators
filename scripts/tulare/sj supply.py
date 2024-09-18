# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:12:12 2024

@author: armen
"""

import pandas as pd
#Reading supply portfolio data (data is for water years)
supply_portfolios = pd.read_csv(r"C:\Users\armen\Downloads\supply_portfolios\supply_portfolios.csv")


sjr_data = supply_portfolios.loc[supply_portfolios.HR=='San Joaquin River']
sjr_data = sjr_data.pivot(index = 'SupplyCategory', columns = 'Year', values = 'taf')
sjr_data = sjr_data.transpose().reset_index()

sjr_data = sjr_data.rename(columns={"Year": "wy"})

#Separating CVP data into local sources and imports
cvp_data = pd.read_csv(r"C:\Users\armen\Downloads\DWR_WaterBalances\cvp_annual_wy.csv")
cvp_data = cvp_data.rename(columns={'year': 'wy'})
cvp_data = cvp_data.rename(columns={'deliveries_wy_weighted': 'cvp_af'})
cvp_data['source'] = 'delta_imports'
cvp_data.loc[(cvp_data.cvp_branch=='Friant-Kern Canal') | (cvp_data.cvp_branch=='Madera Canal and Millerton Lake'), 'source'] = 'local_sj'
cvp_data.loc[(cvp_data.cvp_branch=='Sacramento River') | (cvp_data.cvp_branch=='Tehama-Colusa Canal'), 'source'] = 'local_sac'
cvp_data = cvp_data.groupby(['wy','source', 'hydrologic_region']).sum().reset_index()
sjv_imports = cvp_data.loc[(cvp_data.source=='delta_imports')]
sjv_imports = sjv_imports.loc[sjv_imports.hydrologic_region=='San Joaquin River']
sjv_imports = sjv_imports[['wy','cvp_af']]

sjr_data = sjr_data.merge(sjv_imports, on = 'wy')
sjr_data['cvp_af'] = 0.001*sjr_data.cvp_af
sjr_data['extra_federal'] = sjr_data.Federal - sjr_data.cvp_af
sjr_data['final_local'] = sjr_data.extra_federal + sjr_data.LocalSupplies
sjr_data['final_imports']=sjr_data.cvp_af + sjr_data.SWP
#%%
sjv_local = cvp_data.loc[(cvp_data.source=='local_sj')]
sjv_local = sjv_local.loc[sjv_local.hydrologic_region=='San Joaquin River']
sjv_local = sjv_local[['wy','cvp_af']]

sjv_local = sjv_local[['wy','cvp_af']].reset_index(drop=True)
sjv_imports = sjv_imports[['wy','cvp_af']].reset_index(drop=True)

sjv_federal = sjv_local.cvp_af + sjv_imports.cvp_af
sjv_federal = 0.001*sjv_federal
