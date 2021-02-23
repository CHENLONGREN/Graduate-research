import numpy as np
from ConvLSTM import ConvLSTM
from data1_initial import matrix, std
from data1_initial import matrix1, std1
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math


def to_input(matrix):
    data = np.empty((139, 5, 7))
    data_c = np.empty((139, 5, 7))

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

    count = 0
    for i in range(139):
        week = 0
        while (week < 5):
            day = 0
            while (day < 7):
                data_c[i][week][day] = matrix[count][2]
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

    cost_data = np.empty((len(data) - 4, 5, 5, 7))

    for i in range(len(cost_data)):
        time_step = 0
        while (time_step < 5):
            cost_data[i][time_step] = data_c[i + time_step]
            time_step = time_step + 1

    print(input_data.shape)
    print(cost_data.shape)
    return input_data, cost_data


batch_size = 1
time_steps = 5
shape = [5, 7]
channels = 1
kernel = [3, 3]
filters = 1
steps = 100

ConvLSTM = ConvLSTM(batch_size=batch_size, timesteps=time_steps, shape=shape, channels=channels, kernel=kernel, filters=filters, learning_rate=0.001)

data = to_input(matrix)
input_data = data[0]
cost_data = data[1]
for j in range(2):
    print("第%d轮训练" %(j+1))
    for i in range(len(input_data) - 1):
        batch_x = np.reshape(input_data[i], [batch_size, time_steps, 5, 7, 1])
        batch_y = np.reshape(input_data[i + 1], [batch_size, time_steps, 5, 7, 1])
        batch_c = np.reshape(cost_data[i], [batch_size, time_steps, 5, 7, 1])
        cost = ConvLSTM.Persistent_opt(batch_x, batch_y, batch_c)
        print(cost)
        data_output = np.reshape(ConvLSTM.output(batch_x, batch_y, batch_c), [175])
        print("R2:", r2_score(np.reshape(batch_y, [175]), data_output * 4))

data1 = to_input(matrix1)
input_data1 = data1[0]
cost_data1 = data1[1]
for j in range(2):
    print("第%d轮训练" %(j+1))
    for i in range(len(input_data) - 1):
        batch_x = np.reshape(input_data1[i], [batch_size, time_steps, 5, 7, 1])
        batch_y = np.reshape(input_data1[i + 1], [batch_size, time_steps, 5, 7, 1])
        batch_c = np.reshape(cost_data1[i], [batch_size, time_steps, 5, 7, 1])
        cost = ConvLSTM.Persistent_opt(batch_x, batch_y, batch_c)
        print(cost)
        data_output1 = np.reshape(ConvLSTM.output(batch_x, batch_y, batch_c), [175])
        print("R2:", r2_score(np.reshape(batch_y, [175]), data_output1 * 4))

# data_result = np.empty((2, 35))
# batch_y1 = np.reshape(batch_y, [5, 5, 7])
# data_result[0] = np.reshape(batch_y1[4], [35])
# for i in range(35):
#     data_result[0][i] = math.log(data_result[0][i])
#
# data_output = ConvLSTM.output(batch_x, batch_y, batch_c)
# data_output = np.reshape(data_output, [5, 5, 7])
# data_result[1] = np.reshape(a[4], [35])
# for i in range(35):
#     data_result[1][i] = math.log(data_result[1][i])
#
# fig, ax = plt.subplots()
# x = a = np.linspace(1, 35, 35)
# ax.plot(x, data_result[0], color='b', label='row')
# ax.plot(x, data_result[1], color='r', label='predict')
# leg = ax.legend()
# plt.show()
