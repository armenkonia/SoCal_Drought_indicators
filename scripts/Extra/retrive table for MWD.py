# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:23:00 2024

@author: armen
"""

import pdfplumber
import pandas as pd
from io import StringIO

doc_path = r"C:\Users\armen\Desktop\Drought Indicators - SoCal\Supplies\SWP\Bulletin 132\2020-urban-water-management-plan-june-2021.pdf"
page_number = 285

with pdfplumber.open(doc_path) as pdf:
    # Get the specific page (0-indexed)
    page = pdf.pages[page_number - 1]
    text = page.extract_text()
    
data_lines = text.splitlines()[6:51]  # Remove first 6 lines
cleaned_data = "\n".join(data_lines)
column_names = ['Calendar Year','Local Supplies','LA Aqueduct','Colorado river Aqueduct','State Water Project', 'Total']
df = pd.read_csv(StringIO(cleaned_data), delimiter=' ', names=column_names)
df.set_index('Year', inplace=True)
df = df.replace(',', '').apply(pd.to_numeric)
