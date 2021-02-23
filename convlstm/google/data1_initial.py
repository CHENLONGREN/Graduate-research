import numpy as np
import pandas as pd
import math


def up_noise(matrix):
    for i in range(len(matrix) - 1):
        if (matrix[i][0] > matrix[len(matrix) - 1][0]):
            matrix[i][0] = matrix[len(matrix) - 1][0]
        if (matrix[i][1] > matrix[len(matrix) - 1][1]):
            matrix[i][1] = matrix[len(matrix) - 1][1]
        if (matrix[i][2] > matrix[len(matrix) - 1][2]):
            matrix[i][2] = matrix[len(matrix) - 1][2]
        if (matrix[i][3] > matrix[len(matrix) - 1][3]):
            matrix[i][3] = matrix[len(matrix) - 1][3]
    return matrix

def down_noise(matrix):
    for i in range(len(matrix) - 1):
        if (matrix[i][0] < matrix[len(matrix) - 1][0]):
            matrix[i][0] = matrix[len(matrix) - 1][0]
        if (matrix[i][1] < matrix[len(matrix) - 1][1]):
            matrix[i][1] = matrix[len(matrix) - 1][1]
        if (matrix[i][2] < matrix[len(matrix) - 1][2]):
            matrix[i][2] = matrix[len(matrix) - 1][2]
        if (matrix[i][3] < matrix[len(matrix) - 1][3]):
            matrix[i][3] = matrix[len(matrix) - 1][3]
    return matrix

def SquaredDistance(x, y):
    sd = np.sum(np.power(np.subtract(x, y), 2.0))
    return sd

data = pd.read_csv("D:\Data\OPT\convlstm\data622861879.csv")
df = pd.DataFrame(data)
# df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=True)
matrix = np.array(df)

data = pd.read_csv("D:\Data\OPT\convlstm\data22.csv")
df = pd.DataFrame(data)
# df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=True)
matrix1 = np.array(df)

matrix = np.delete(matrix, [len(matrix)-2, len(matrix)-3], axis=0)
matrix1 = np.delete(matrix1, [len(matrix1)-2, len(matrix1)-3], axis=0)

std = matrix[len(matrix)-1]
matrix = np.delete(up_noise(matrix), [len(matrix)-1], axis=0)
# matrix = np.delete(down_noise(matrix), [len(matrix)-1], axis=0)

std1 = matrix1[len(matrix1)-1]
matrix1 = np.delete(up_noise(matrix1), [len(matrix1)-1], axis=0)
# matrix1 = np.delete(down_noise(matrix1), [len(matrix1)-1], axis=0)

# matrix = matrix[:, 3]
# matrix1 = matrix1[:, 3]
# print(math.log(SquaredDistance(matrix, matrix1)))


