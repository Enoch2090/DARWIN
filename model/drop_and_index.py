import pandas as pd
import os, re, sys

df = pd.read_json('./data/data.json', lines=True)
df.drop(columns=['drive_train', 'fuel_type', 'transmission', 'mpg', 'price', 'engine', 'interior_color', 'turbo'], inplace=True)
df['make'] = df['contents'].apply(lambda x: re.findall(r'\d{4} [A-Z]\w+', x)[0].split()[1])
df.to_json('./data_dropped/data_dropped.json', orient='records', lines=True)
exec_path = sys.executable
os.system(f"{exec_path} -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 4 -input ./data_dropped -index ./index -storePositions -storeDocvectors -storeRaw")
