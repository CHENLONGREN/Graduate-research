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
channels = 1
kernel = [2, 2]
filters = 1
steps = 1000

ConvLSTM = ConvLSTM(batch_size=batch_size, timesteps=timesteps, shape=shape, channels=channels, kernel=kernel, filters=filters, learning_rate=0.001)

# data3是去除了不需要的属性的数据，形状是（len， 168， 6）
data3 = np.empty((len(data2), 168, 4))
for i in range(len(data2)):
    data3[i] = np.delete(data2[i], [0, 1, 2, 3, 5], axis=1)
    data3[:, [0, 1, 2, 3]] = data3[:, [3, 2, 1, 0]]

input_data = np.empty((len(data3), 6, 4, 7, 1))

for i in range(len(data3)):
    t = 0
    for j in range(6):
        temp = np.empty(28)
        for m in range(28):
            temp[m] = data3[i][t][3]
            t += 1
        input_data[i][j] = np.reshape(temp, [4, 7, 1])

y_click = np.empty((steps, 2))
for i in range(steps):
    batch = get_random_block_from_data(input_data, batch_size)
    batch_x = np.delete(batch, [3, 4, 5], axis=1)
    batch_y = np.delete(batch, [0, 1, 2], axis=1)
    cost = ConvLSTM.opt(batch_x, batch_y)
    y_click[i][0] = i+1
    y_click[i][1] = math.log(cost)
    # print(cost)

# np.savetxt('y_click.csv', y_click, delimiter = ',')
print(cost)
