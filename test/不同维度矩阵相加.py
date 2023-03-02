import torch

A= torch.rand(1,1,325,64)
B = torch.rand(16,24,1,64)
print((A+B).shape)