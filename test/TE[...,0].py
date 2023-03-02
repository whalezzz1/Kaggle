import torch
import torch.nn.functional as F

TE = torch.rand(16, 24, 2)

for i in range(TE.shape[0]):
    A = TE[..., 0][i]
    B = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)


print(TE, TE.shape, '\n', B, B.shape)
