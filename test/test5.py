import torch.nn.functional as F

units = 8
input_dims = 16
activations = F.relu
if isinstance(units, int):
    units = [units]
    input_dims = [input_dims]
    activations = [activations]
    print(units, input_dims, activations)
