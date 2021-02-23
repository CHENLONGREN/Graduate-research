import numpy as np
from ConvLSTM import ConvLSTM
from data1_initial import matrix, std
from data1_initial import matrix1, std1
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def to_input(matrix):
    data = np.empty((139, 5, 7))

    count = 0
    for i in range(139):
        week = 0
        while (week < 5):
            day = 0
            while (day < 7):
                data[i][week][day] = matrix[count][0]
                day = day + 1
                count = count + 1
            week = week + 1
        count = count - 28

    input_data = np.empty((len(data) - 4, 5, 5, 7))

    for i in range(len(input_data)):
        time_step = 0
        while (time_step < 5):
            input_data[i][time_step] = data[i + time_step]
            time_step = time_step + 1

    print(input_data.shape)
    return input_data


batch_size = 1
time_steps = 5
shape = [5, 7]
channels = 1
kernel = [3, 3]
filters = 1
steps = 100

ConvLSTM = ConvLSTM(batch_size=batch_size, timesteps=time_steps, shape=shape, channels=channels, kernel=kernel, filters=filters, learning_rate=0.001)

input_data = to_input(matrix)
for j in range(2):
    print("第%d轮训练" %(j+1))
    for i in range(len(input_data) - 1):
        batch_x = np.reshape(input_data[i], [batch_size, time_steps, 5, 7, 1])
        batch_y = np.reshape(input_data[i + 1], [batch_size, time_steps, 5, 7, 1])
        cost = ConvLSTM.opt(batch_x, batch_y)
        print(cost)
        data_output = np.reshape(ConvLSTM.output1(batch_x, batch_y), [175])
        print("R2:", r2_score(np.reshape(batch_y, [175]), data_output*3))

input_data1 = to_input(matrix1)
for j in range(2):
    print("第%d轮训练" %(j+3))
    for i in range(len(input_data1) - 1):
        batch_x = np.reshape(input_data1[i], [batch_size, time_steps, 5, 7, 1])
        batch_y = np.reshape(input_data1[i + 1], [batch_size, time_steps, 5, 7, 1])
        cost = ConvLSTM.opt(batch_x, batch_y)
        print(cost)
        data_output1 = np.reshape(ConvLSTM.output1(batch_x, batch_y), [175])
        print("R2:", r2_score(np.reshape(batch_y, [175]), data_output1*3))



