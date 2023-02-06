from tensorflow.keras.models import load_model

from data import get_test_data
import numpy as np
import matplotlib.pyplot as plt

x, y, episode_duration = get_test_data()
# print(x[0][0])
# print(x[1][0])
# print(x[episode_duration-1][0])
# print(y[0])
# exit()
y = np.array(y)

model = load_model('model')

acc = []
for i in range(episode_duration):
    episodes = np.array(x[i])
    print(episodes.shape)
    yp = model.predict(episodes)
    acc.append(np.mean(np.abs(y - yp), axis=0))

x_err = []
y_err = []

for a in acc:
    x_err.append(a[0])
    y_err.append(a[1])
plt.plot(x_err)
plt.savefig('x.png')
plt.close()

plt.plot(y_err)
plt.savefig('y.png')
plt.close()
