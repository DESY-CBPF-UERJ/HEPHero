import requests
import urllib.request 
import pandas as pd  
import numpy as np
from tqdm import tqdm
import json


url='https://lhapdf.hepforge.org/pdfsets'  
html = requests.get(url).content
df_list = pd.read_html(html)
df_LHA = df_list[-1]
del df_LHA["Set name and links.1"]
del df_LHA["Set name and links.2"]
del df_LHA["Latest data version"]
del df_LHA["Notes"]

error_type_column = []
for name in tqdm(df_LHA["Set name and links"]):
    url_type="https://lhapdfsets.web.cern.ch/current/" + name + "/" + name + ".info"
    error_type = "none"
    for line in urllib.request.urlopen(url_type):
        line = line.decode('utf-8')[:-1]
        if line[:9] == "ErrorType":
            error_type = line.split(":")[1].strip()
    error_type_column.append(error_type)

pdf_type_column = error_type_column[:]
pdf_type_column = ['hessian' if 'hessian' in pdf_type else pdf_type for pdf_type in pdf_type_column]
pdf_type_column = ['mc' if 'replicas' in pdf_type else pdf_type for pdf_type in pdf_type_column]
pdf_type_column = ['unknown' if (pdf_type != "hessian" and pdf_type != "mc") else pdf_type for pdf_type in pdf_type_column]
df_LHA["PDF_Type"] = pdf_type_column
#df_LHA.to_csv('LHAPDF.csv', index=False)

pdf_dict = dict(zip(df_LHA["LHAPDF ID"], df_LHA["PDF_Type"]))
with open("pdf_type.json", 'w') as pdf_file:
    json.dump(pdf_dict, pdf_file)
