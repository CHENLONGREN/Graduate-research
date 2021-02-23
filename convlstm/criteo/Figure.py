import numpy as np
import matplotlib.pyplot as plt


y_pv = np.loadtxt(open("D:/Program/convlstm/y_pv.csv","rb"),delimiter=",",skiprows=0)
list_matrix = y_pv.tolist()
y_pv = np.array(list_matrix)
y_cv = np.loadtxt(open("D:/Program/convlstm/y_cv.csv","rb"),delimiter=",",skiprows=0)
list_matrix = y_cv.tolist()
y_cv = np.array(list_matrix)
y_click = np.loadtxt(open("D:/Program/convlstm/y_click.csv","rb"),delimiter=",",skiprows=0)
list_matrix = y_click.tolist()
y_click = np.array(list_matrix)
y_total = np.loadtxt(open("D:/Program/convlstm/ y_total.csv","rb"),delimiter=",",skiprows=0)
list_matrix = y_total.tolist()
y_total = np.array(list_matrix)

x = np.linspace(0, 1000, 1000)

fig, ax = plt.subplots()
ax.plot(x, y_pv, color='b', label='pv')
ax.plot(x, y_cv, color='g', label='cv')
ax.plot(x, y_click, color='y', label='click')
ax.plot(x, y_total, color='r', label='total')
leg = ax.legend()
plt.show()
