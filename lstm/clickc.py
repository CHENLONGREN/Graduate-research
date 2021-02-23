import numpy as np
from lstm import LSTM
import pandas as pd
from sklearn.metrics import r2_score


def to_input(matrix):
    input_data = np.empty((len(matrix) - 34, 35))

    count = 0
    for i in range(len(input_data)):
        time_step = 0
        while (time_step < 35):
            input_data[i][time_step] = matrix[count][1]
            time_step = time_step + 1
            count = count + 1
        count = count - 34

    cost_data = np.empty((len(matrix) - 34, 35))

    count = 0
    for i in range(len(cost_data)):
        time_step = 0
        while (time_step < 35):
            cost_data[i][time_step] = matrix[count][2]
            time_step = time_step + 1
            count = count + 1
        count = count - 34

    return input_data, cost_data

a = np.arange(980, 1004)

data = pd.read_csv("D:\Data\OPT\convlstm\data22.csv")
df = pd.DataFrame(data)
# df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=True)
matrix = np.array(df)
matrix = np.delete(matrix, a, axis=0)

data1 = pd.read_csv("D:\Data\OPT\convlstm\data648028230.csv")
df1 = pd.DataFrame(data1)
# df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=True)
matrix1 = np.array(df1)
matrix1 = np.delete(matrix1, a, axis=0)

batch_size = 1
time_steps = 35

LSTM = LSTM(input_size=1, state_size=10, hidden_sum=1, output_size=1, time_steps=35, batch_size=batch_size,
            learning_rate=0.001)

data = to_input(matrix)
input_data = data[0]
cost_data = data[1]
for j in range(2):
    print("第%d轮训练" %(j+1))
    for i in range(len(input_data) - 1):
        batch_x = np.reshape(input_data[i], [batch_size, time_steps, 1])
        batch_y = np.reshape(input_data[i + 1], [batch_size, time_steps])
        batch_c = np.reshape(cost_data[i], [batch_size, time_steps])
        cost = LSTM.Persistent_opt(batch_x, batch_y, batch_c)
        print(cost)
        data_output = np.reshape(LSTM.output(batch_x, batch_y, batch_c), [35])
        print("R2:", r2_score(np.reshape(batch_y, [35]), data_output))

data1 = to_input(matrix1)
input_data1 = data1[0]
cost_data1 = data1[1]
for j in range(2):
    print("第%d轮训练" %(j+1))
    for i in range(len(input_data1) - 1):
        batch_x = np.reshape(input_data1[i], [batch_size, time_steps, 1])
        batch_y = np.reshape(input_data1[i + 1], [batch_size, time_steps])
        batch_c = np.reshape(cost_data1[i], [batch_size, time_steps])
        cost = LSTM.Persistent_opt(batch_x, batch_y, batch_c)
        print(cost)
        data_output1 = np.reshape(LSTM.output(batch_x, batch_y, batch_c), [35])
        print("R2:", r2_score(np.reshape(batch_y, [35]), data_output1))

