# Fei

import pandas as pd

# Load file
f = open('en-fr.csv', 'rb')

# Write file
new_f = open('eng-fra.csv', 'wb')

# EDA
for i in range(5000):
    line = f.readline()
    new_f.write(line)
    print(line)

new_f.close()
f.close()

df = pd.read_csv('en-fr.csv',nrows=2000, encoding='utf-8')
print(df.head(10))
output = df.values.tolist()
print(output)

pass