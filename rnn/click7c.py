import numpy as np
from RNN import RNN
import pandas as pd


# def get_random_block_from_data(data, times_steps, batch_size):
#     start_index = np.random.randint(0, len(data) - times_steps - batch_size)
#     batch_x = data[start_index: start_index + batch_size]
#     batch_y = data[start_index + times_steps: start_index + times_steps + batch_size]
#     return batch_x, batch_y


data = pd.read_csv("D:\Data\OPT\Rnn\data11.csv")
df = pd.DataFrame(data)
df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=True)
matrix = np.array(df)

matrix = np.delete(matrix, [len(matrix)-1, len(matrix)-2], axis=0)

input_data = np.empty((len(matrix) - 6, 7))

count = 0
for i in range(len(input_data)):
    time_step = 0
    while (time_step < 7):
        input_data[i][time_step] = matrix[count][1]
        time_step = time_step + 1
        count = count + 1
    count = count - 6

cost_data = np.empty((len(matrix) - 6, 7))

count = 0
for i in range(len(cost_data)):
    time_step = 0
    while (time_step < 7):
        cost_data[i][time_step] = matrix[count][2]
        time_step = time_step + 1
        count = count + 1
    count = count - 6

training_epoch = 1000
batch_size = 1
time_steps = 7

RNN = RNN(input_size=1, state_size=10, hidden_sum=1, output_size=1, time_steps=7, batch_size=batch_size,
          learning_rate=0.001)

i = 0
for i in range(len(input_data)-1):
    batch_x = np.reshape(input_data[i], [batch_size, time_steps, 1])
    batch_y = np.reshape(input_data[i+1], [batch_size, time_steps])
    batch_c = np.reshape(cost_data[i], [batch_size, time_steps])
    cost = RNN.Persistent_opt(batch_x, batch_y, batch_c)
    print(cost)