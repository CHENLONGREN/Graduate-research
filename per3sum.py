import numpy as np
import pandas as pd

data = pd.read_csv("D:\Data\OPT\cp622861879.csv")
df = pd.DataFrame(data)
df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=True)
matrix = np.array(df)

data = np.empty([1003, 4])

i = 0
t = 0
while (i<len(matrix)):
    for j in range(4):
        data[t][j] = np.sum(matrix[i:i+3][:, j])
    i = i + 3
    t = t + 1

np.savetxt('data622861879.csv', data, delimiter = ',')

