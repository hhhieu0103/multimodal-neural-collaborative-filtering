import pandas as pd
import glob, re

csv_files = glob.glob('Reviews/*.csv')

df_list = []
for file in csv_files:
    print(f'Reading {file}')
    df = pd.read_csv(file)
    pattern = r'Review(\d+).csv'
    match = re.search(pattern, file)
    appid = match.group(1)
    df['app_id'] = appid
    df_list.append(df)

final_df = pd.concat(df_list, ignore_index=True)
final_df.to_csv('Reviews.csv')