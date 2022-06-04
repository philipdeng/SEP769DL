# Fei

import pandas as pd
from tqdm import tqdm

# Load file
f = open('en-fr.csv', 'rb')

# Write file
new_f = open('eng-fra.csv', 'wb')

# Select
i = 0
while i < 1000000:
    line = f.readline()
    if line.decode("utf-8").count(",") == 1:
        new_f.write(line)
        i += 1

new_f.close()
f.close()

# df = pd.read_csv('en-fr.csv',nrows=2000, encoding='utf-8')
# print(df.head(10))
# output = df.values.tolist()
# print(output)

pass