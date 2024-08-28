# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:46:42 2024

@author: armen
"""

import pdfplumber
import time
import pandas as pd
from PyPDF2 import PdfReader
import re

doc_path = r"C:\Users\armen\Desktop\Drought Indicators - SoCal\Main\Colorado\Extra\Water Accounting Reports\2015.pdf"
page_number = 54

all_tables = []
start_time = time.time()
with pdfplumber.open(doc_path) as pdf:
        page = pdf.pages[page_number - 1]  # Page numbers are 0-indexed in pdfplumber
        tables = page.extract_tables()

data_str = tables[0][4][0]
# Split into rows
rows = [line.strip() for line in data_str.strip().split('\n')]

# Split each row into columns
data = [re.split(r'\s+', row) for row in rows]

# Create DataFrame
df = pd.DataFrame(data)

# Optionally set column names if known or provide generic names
df.columns = [f'Column_{i+1}' for i in range(df.shape[1])]

#%%
# Function to shift each row based on the number of leading `None` values
def shift_row(row):
    # Count leading None values
    leading_none_count = len(row) - len([x for x in row if x is not None])
    # Shift row to the right by leading_none_count
    return [None]*leading_none_count + [x for x in row if x is not None]

# Apply the function to each row
df_shifted = pd.DataFrame(df.apply(shift_row, axis=1).tolist(), columns=df.columns)
