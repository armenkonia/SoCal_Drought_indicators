# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:20:59 2024

@author: armen
"""
    
# page = pdf.pages[0]
# im = page.to_image()
# im.debug_tablefinder()

# table_settings = {
#     "vertical_strategy": "text",
#     "horizontal_strategy": "text",
#     "snap_y_tolerance": 5,
#     "intersection_x_tolerance": 15,
# }
# im.reset().debug_tablefinder(table_settings)

# table_1 = all_tables[0][1][0]
# pages_of_interest = [420, 421, 422, 423, 425, 426, 427, 428, 430, 431, 434, 435, 436]

# filtered_tables = []
# for page_number in pages_of_interest:
#         page = pdf.pages[page_number - 1]  # Page numbers are 0-indexed in pdfplumber
#         tables = page.extract_tables()
#         filtered_tables.append((page_number, tables))

#%%
import pdfplumber
import time

doc_path = r"C:\Users\armen\Desktop\Drought Impacts\SoCal Indicators\Main\Bulletin 132-19 020323.pdf"
start_page = 418
end_page = 438

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
doc_path = r"C:\Users\armen\Desktop\Drought Impacts\SoCal Indicators\Main\Bulletin 132-19 020323.pdf"
page_number = 379 + 44

with pdfplumber.open(doc_path) as pdf:
    # Get the specific page (0-indexed)
    page = pdf.pages[page_number - 1]
    text = page.extract_text()

# print(text)
#%%
page = pdf.pages[392 + 44 - 1]

table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "text",
    "snap_y_tolerance": 5,
    "intersection_x_tolerance": 15,
}

tables_1 = page.extract_tables(table_settings)
table_1 = tables_1[0]
im = page.to_image()
im.debug_tablefinder(table_settings)
#%%
import pandas as pd
table_1 = table_1[7:64]
df = pd.DataFrame(table_1)
df = df[0].str.replace(',', '').str.split(expand=True)

df = df[0].str.split(expand=True)

df.apply(pd.to_numeric)
#%%
df_parts_df.to_csv(r"C:\Users\armen\Desktop\trial.csv")
new_df = pd.read_csv(r"C:\Users\armen\Desktop\trial.csv",index_col=0)
