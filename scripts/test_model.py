from tensorflow.keras.models import load_model
import numpy as np

def read_data():
    lines = open('inp-tst').read().split('\n')
    n = -1
    inp = []
    k = [52.5, 34, 30, 3, 3, 30, 180, 30]
    
    for line in lines:
        if line.startswith('---'):
            continue
        
        n += 1
        data = float(line.split(':')[-1])
        data /= k[n%8]
        inp.append(data)
    
    return np.array(inp)
        

model = load_model('res/model-dnn-512-256-128-64-32-relu-relu-relu-relu-relu-adam-mse-64')
x = read_data()
print(x.shape)
x = x.reshape(1, -1)
y = model.predict(x)

for i in range(11):
    print(f'{i}: ({y[0,i*2]}, {y[0,i*2+1]})')