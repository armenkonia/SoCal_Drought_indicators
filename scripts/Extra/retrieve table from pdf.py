# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:27:00 2024

@author: armen
"""
import pandas as pd
from PyPDF2 import PdfReader

doc_path = r"..\Main\Bulletin 132-19 020323.pdf"
page_number = 436
with open(doc_path, "rb") as f:
    pdf = PdfReader(f)
    page = pdf.pages[page_number - 1]
    text = page.extract_text()
lines = text.split('\n')
filtered_lines = lines[9:66]
table_data = []
for line in filtered_lines:
    values = line.split() # Split the line by whitespace to get individual values
    table_data.append(values)
    
df = pd.DataFrame(table_data)
df.columns = ['Year','AVEK','AVEK','Santa Clarita', 'Ventura', 'Coachella', 'Desert', 'Metropolitan' ]
df = df.apply(lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))
df = df.apply(pd.to_numeric, errors='coerce')
#%%

page_number = 437
with open(doc_path, "rb") as f:
    pdf = PdfReader(f)
    page = pdf.pages[page_number - 1]
    text = page.extract_text()
lines = text.split('\n')
n = 13
filtered_lines = lines[n:n+57]
table_data = []
for line in filtered_lines:
    values = line.split() # Split the line by whitespace to get individual values
    table_data.append(values)
    
df = pd.DataFrame(table_data)
df.columns = ['Year','San Bernandino','Santa Barbara','Santa Clarita', 'Ventura', 'Avek', 'Dudley Ridge', 'Municipal and Industrial', 'Agriculture', 'Kings' ]
df = df.apply(lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))
df = df.apply(pd.to_numeric, errors='coerce')
#%%
page_number = 379+44
with open(doc_path, "rb") as f:
    pdf = PdfReader(f)
    page = pdf.pages[page_number - 1]
    text = page.extract_text()
lines = text.split('\n')
n = 0
table_data = []
for line in lines:
    values = line.split()
    table_data.append(values)
    
df = pd.DataFrame(table_data)

# Find the index of the row where the first column value is 1962
start_index = df[df[0] == '1962'].index[0]
# Slice the DataFrame from this index onwards
df = df.iloc[start_index:start_index + 57].reset_index(drop=True)


# df.columns = ['Year','Desert', 'Metropolitan', 'Coachella', 'Metropolitan', 'San Bernardino', 'San Bernardino', 'San Bernardino', 'San Gorgonio', 'San Gorgonio']
df = df.apply(lambda col: col.str.replace(',', '')
                        .str.replace('(', '-')
                        .str.replace(')', '')
                        .apply(pd.to_numeric, errors='coerce'))
# df = df.apply(lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))
# df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')

#%%
import pandas as pd
from PyPDF2 import PdfReader

doc_path = r"..\Main\Bulletin 132-19 020323.pdf"
start_page = 376 + 44
end_page = 386 + 44
combined_df = pd.DataFrame()

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

    # Find the index of the row where the first column value is 1962
    start_index = df[df[0] == '1962'].index[0]
    # Slice the DataFrame from this index onwards
    df = df.iloc[start_index:start_index + 57].reset_index(drop=True)

    # Process the data
    df = df.apply(lambda col: col.str.replace(',', '')
                              .str.replace('(', '-')
                              .str.replace(')', '')
                              .apply(pd.to_numeric, errors='coerce'))

    # Remove all empty columns
    df = df.dropna(axis=1, how='all')

    # Combine the data
    combined_df = pd.concat([combined_df, df], ignore_index=True,axis=1)

# print(combined_df)
