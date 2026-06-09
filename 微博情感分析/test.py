# check_data.py

import pandas as pd

df = pd.read_csv("weibo_senti_100k.csv")

print(df.head())
print(df.columns)
print(df.shape)