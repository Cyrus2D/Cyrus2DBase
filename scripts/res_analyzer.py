import numpy as np
import matplotlib.pyplot as plt

edp = np.genfromtxt('res/edp-dnn-128-64-relu-relu-adam-mse-64', delimiter=',')

err_range = [
    0,
    0.1,
    0.3,
    0.4,
    0.6,
    0.8,
    0.9,
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    100,
]

counter = []
for i in range(len(err_range) - 1):
    print((edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]))
    counter.append(np.sum(np.where((edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]), 1, 0)))

counter.append(0)
print(counter)
plt.plot(err_range, counter)
plt.show()
