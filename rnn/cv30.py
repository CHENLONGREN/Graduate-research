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

matrix = np.delete(matrix, matrix[990:1002], axis=0)

input_data = np.empty((len(matrix) - 29, 30))

count = 0
for i in range(len(input_data)):
    time_step = 0
    while (time_step < 30):
        input_data[i][time_step] = matrix[count][3]
        time_step = time_step + 1
        count = count + 1
    count = count - 29

training_epoch = 1000
batch_size = 1
time_steps = 30

RNN = RNN(input_size=1, state_size=10, hidden_sum=1, output_size=1, time_steps=30, batch_size=batch_size,
          learning_rate=0.001)

i = 0
for i in range(len(input_data)-1):
    batch_x = np.reshape(input_data[i], [batch_size, time_steps, 1])
    batch_y = np.reshape(input_data[i+1], [batch_size, time_steps])
    cost = RNN.opt(batch_x, batch_y)
    print(cost)