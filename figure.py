import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("D:/Data/OPT/convlstm/data622861879.csv","rb"),delimiter=",",skiprows=0)
y = np.array(data)

x = np.linspace(0, 1003, 1003)

fig, ax = plt.subplots()
ax.plot(x, y[:, 3], color='b', label='cv')
leg = ax.legend()
plt.show()
