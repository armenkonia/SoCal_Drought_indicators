# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:09:24 2024

@author: armen
"""

import pdfplumber
import time
import pandas as pd
from PyPDF2 import PdfReader
# doc_path = r"..\Main\Bulletin 132-19 020323.pdf"
doc_path = r"C:\Users\armen\Desktop\B132-23 Appendix B 041724.pdf"
start_page = 42 + 10
end_page = start_page + 19

all_tables = []
start_time = time.time()
with pdfplumber.open(doc_path) as pdf:
    for page_number in range(start_page, end_page + 1):
        page = pdf.pages[page_number - 1]  # Page numbers are 0-indexed in pdfplumber
        tables = page.extract_tables()
        all_tables.append((page_number, tables))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to extract tables from pages {start_page} to {end_page}: {elapsed_time:.2f} seconds")
#%%
all_dfs = []
for i in range(len(all_tables)):
    page_data = all_tables[i][1][0]
    header_rows = page_data[:6]
    header_df = pd.DataFrame(header_rows).fillna('')
    year_row_index = int(header_df[header_df[0].str.contains('1962', na=False)].index[0])
    header_df = header_df.iloc[:year_row_index]
    all_dfs.append(header_df)
combined_df_headers = pd.concat(all_dfs, ignore_index=True,axis=1)
combined_df_headers = combined_df_headers.apply(lambda col: col.str.replace('\n', ' ', regex=False) if col.dtype == 'object' else col)
#%%
combined_df_data = pd.DataFrame()
for page_number in range(start_page, end_page + 1):
    with open(doc_path, "rb") as f:
        pdf = PdfReader(f)
        page = pdf.pages[page_number - 1]
        text = page.extract_text()

    lines = text.split('\n')
    table_data = []
    for line in lines:
        values = line.split()
        table_data.append(values)

    df = pd.DataFrame(table_data)
    start_index = df[df[0] == '1962'].index[0]
    df = df.iloc[start_index:start_index + 61].reset_index(drop=True)
    df = df.apply(lambda col: col.str.replace(',', '')
                              .str.replace('(', '-')
                              .str.replace(')', '')
                              .apply(pd.to_numeric, errors='coerce'))
    df = df.dropna(axis=1, how='all')
    combined_df_data = pd.concat([combined_df_data, df], ignore_index=True,axis=1)

meta_df = pd.concat([combined_df_headers,combined_df_data], ignore_index=True)
meta_df.index = meta_df[0]
meta_df.columns = meta_df.iloc[0]  # Set the first row as header
meta_df = meta_df.loc[:, meta_df.columns != "Calendar Year"]

