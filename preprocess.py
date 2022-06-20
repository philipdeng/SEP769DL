# Fei

import pandas as pd
from tqdm import tqdm
import random

# Load file
f = open('en-fr.csv', 'rb')

# Write file
new_f = open('eng-fra.csv', 'wb')

# Select
i = 0
while i < 2000000:
    if random.randint(1,10) <5:
        continue
    line = f.readline()
    if line.decode("utf-8").count(",") == 1:
        new_f.write(line)
        i += 1

new_f.close()
f.close()

print("extracted")

# df = pd.read_csv('en-fr.csv',nrows=2000, encoding='utf-8')
# print(df.head(10))
# output = df.values.tolist()
# print(output)

pass