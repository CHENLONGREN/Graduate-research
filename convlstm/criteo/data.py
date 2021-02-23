import pandas as pd
import numpy as np

data = pd.read_csv("D:\Data\OPT\criteo000.csv", names = list(range(1, 69)))
df = pd.DataFrame(data)
df = df.iloc[:,[1,4,6,17,18,19,20,67]]   # 选中列
df.columns = ['day_of_week', 'cp_id', 'ag_id', 'imps','clicks','cost','cv','rp_date']
order = ['ag_id','cp_id', 'cv', 'imps', 'clicks', 'cost', 'rp_date', 'day_of_week']
df = df[order]

df = df[~df['ag_id'].isin([0])]   # 删除某列中包含某一值的行
df = df[~df['ag_id'].isin([-1])]

df_data = list(df.groupby(['ag_id', 'cp_id']))   # DataFrame的list化

count = 0
for i in range(len(df_data)):
    j = i-count
    if(len(set(df_data[j][1]['rp_date'])) != 164):
            del df_data[j]
            count+=1

for i in range(len(df_data)):
    df_data[i][1].sort_values(by='rp_date', inplace=True)   # 按日期排序

# data是初始数据，形状是（len， 164， 7）
data = np.empty((len(df_data), 164, 7))
date = df_data[0][1][['rp_date']]   # 选取将dataframe中的rp_date属性

for i in range(len(df_data)):
    df_data[i][1].drop(df.columns[[6]], axis=1, inplace=True)   # 将原dataframe中的rp_date属性删除，因为只有rp_date是字符类型数据
    data[i] = np.array(df_data[i][1]).tolist()   # dataframe数组化

# data1是对行进行补全后的数据，形状是（len， 168， 7）
data1 = np.empty((len(data), 168, 7))

for i in range(len(data)):
    temp = np.empty((4, 7))
    temp[0][2] = temp[1][2] = temp[2][2] = temp[3][2] = np.mean(data[i][:, 2])
    temp[0][3] = temp[1][3] = temp[2][3] = temp[3][3] = np.mean(data[i][:, 3])
    temp[0][4] = temp[1][4] = temp[2][4] = temp[3][4] = np.mean(data[i][:, 4])
    data1[i] = np.insert(data[i], 164, values=temp, axis=0)

# data2是对列进行补全后的数据，形状是（len， 168， 9）
data2 = np.empty((len(data1), 168, 9))

week = np.ones(7)
week4 = week
for i in range(3):
    week = week + 1
    week4 = np.concatenate((week4, week), axis=0)

month = np.ones(28)
time = month
high = week4
for i in range(5):
    month = month + 1
    time = np.concatenate((time, month), axis=0)
    high = np.concatenate((high, week4), axis=0)

high_time = np.empty((168, 2))
high_time[:, 0] = high
high_time[:, 1] = time

for i in range(len(data1)):
    data2[i] = np.insert(data1[i], 7, values=high_time.T, axis=1)  # insert()函数添加列时，需要将第二个数组转置。可以直接用append()添加列。
