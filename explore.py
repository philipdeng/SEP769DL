# Fei

import pandas as pd

# Load file
f = open('en-fr.csv', 'rb')

# EDA
for i in range(5):
    line = f.readline()
    print(line)

df = pd.read_csv('en-fr.csv',nrows=2000, encoding='utf-8')
print(df.head(10))
output = df.values.tolist()
print(output)

pass