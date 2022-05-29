# Fei

import pandas as pd

f = open('en-fr.csv', 'rb')

for i in range(5):
    line = f.readline()
    print(line)

df = pd.read_csv('en-fr.csv',nrows=1000, encoding='utf-8')
df.head(10)

pass