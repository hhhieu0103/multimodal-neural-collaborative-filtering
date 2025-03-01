import pandas as pd
import json
import glob

json_files = glob.glob("Metadata/*.json")

data_list = []

for file in json_files:
    print(f'Reading {file}')
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data_list.extend(data)

df = pd.DataFrame(data_list)

df.to_csv("Final/Metadata-raw.csv", index=False)
