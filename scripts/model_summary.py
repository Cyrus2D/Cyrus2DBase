from tensorflow.keras.models import load_model

model = load_model('res/model-dnn-512-256-128-64-32-relu-relu-relu-relu-relu-adam-mse-64')
model.summary()


