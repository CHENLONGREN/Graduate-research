import numpy as np
from lstm import LSTM
import pandas as pd
from sklearn.metrics import r2_score


def input_data(matrix):
    input_data = np.empty((len(matrix) - 29, 30))

    count = 0
    for i in range(len(input_data)):
        time_step = 0
        while (time_step < 30):
            input_data[i][time_step] = matrix[count][3]
            time_step = time_step + 1
            count = count + 1
        count = count - 29

    return input_data


data = pd.read_csv("D:\Data\OPT\convlstm\data22.csv")
df = pd.DataFrame(data)
# df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=True)
matrix = np.array(df)

a = np.arange(980, 1004)
matrix = np.delete(matrix, a, axis=0)

batch_size = 1
time_steps = 30

LSTM = LSTM(input_size=1, state_size=10, hidden_sum=1, output_size=1, time_steps=30, batch_size=batch_size,
          learning_rate=0.001)

input_data = input_data(matrix)
i = 0
for i in range(len(input_data)-1):
    batch_x = np.reshape(input_data[i], [batch_size, time_steps, 1])
    batch_y = np.reshape(input_data[i+1], [batch_size, time_steps])
    cost = LSTM.opt(batch_x, batch_y)
    print(cost)
    data_output = np.reshape(LSTM.output(batch_x, batch_y, batch_c), [30])
    print("R2:", r2_score(np.reshape(batch_y, [175]), data_output * 3))
    # print("R2:", 1 - (cost / np.var(batch_y)))