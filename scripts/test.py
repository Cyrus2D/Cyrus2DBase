from tensorflow.keras.models import load_model

from data import get_test_data
import numpy as np


x, y, episode_duration = get_test_data()
print(x[0][0])
print(y[0])
exit()
x = np.array(x)
y = np.array(y)

model = load_model('model')

model.predict()

model.save('model')