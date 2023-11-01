import torch

from pytorch_sample import TorchLSTM

# torch.load('./res/torch/model')

model = TorchLSTM([512, 512], ['relu', 'relu'])

model.load_state_dict(torch.load('/data1/aref/2d/Cyrus2DBase/scripts/res/torch/model'))
model.eval()
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('/data1/aref/2d/Cyrus2DBase/scripts/res/torch/lstm') # Save
