from tensorflow.keras.models import load_model

from data import get_test_data
import numpy as np


x, y, episode_duration = get_test_data()
print(x[0][0])
print(y[0])

y = np.array(y)

model = load_model('model')

for i in range(episode_duration):
    yp = model.predict_on_batch(x[i])
    print(y - yp)


model.save('model')