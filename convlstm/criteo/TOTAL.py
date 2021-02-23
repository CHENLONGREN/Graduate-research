import math
import numpy as np
from ConvLSTM import ConvLSTM
from data import data2


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: start_index + batch_size]


batch_size = 32
timesteps = 3
shape = [4, 7]
channels = 3
kernel = [2, 2]
filters = 3
steps = 1000

ConvLSTM = ConvLSTM(batch_size=batch_size, timesteps=timesteps, shape=shape, channels=channels, kernel=kernel, filters=filters, learning_rate=0.001)

# data3是去除了不需要的属性的数据，形状是（len， 168， 6）
data3 = np.empty((len(data2), 168, 6))
for i in range(len(data2)):
    data3[i] = np.delete(data2[i], [0, 1, 5], axis=1)
    data3[:, [0, 1, 2, 3, 4, 5]] = data3[:, [5, 4, 3, 0, 1, 2]]

input_data = np.empty((len(data3), 6, 4, 7, 3))

for i in range(len(data3)):
    t = 0
    for j in range(6):
        temp = np.empty(84)
        for m in range(28):
            temp[m] = data3[i][t][3]
            temp[m + 28] = data3[i][t][4]
            temp[m + 28 + 28] = data3[i][t][5]
            t += 1
        input_data[i][j] = np.reshape(temp, [4, 7, 3])

y_total = np.empty((1000, 2))
t = 0
for i in range(steps):
    batch = get_random_block_from_data(input_data, batch_size)
    batch_x = np.delete(batch, [3, 4, 5], axis=1)
    batch_y = np.delete(batch, [0, 1, 2], axis=1)
    cost = ConvLSTM.opt(batch_x, batch_y)
    if(i%3 == 0):
        y_total[t][0] = t + 1
        y_total[t][1] = math.log(cost)
        t = t + 1
    print(cost)

# np.savetxt(' y_total.csv',  y_total, delimiter = ',')
# print(cost)
