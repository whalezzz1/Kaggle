import torch
import torch.nn.functional as F


def attention_f(QL, KL, VL, d, K, batch_size):

    QL = torch.cat(torch.split(QL, K, dim=-1),
                   dim=0)  # 将(16,1,325,64)分成8块torch.Size([16, 1, 325, 8])再在第一维拼起来得到（[128, 1, 325, 8]）
    KL = torch.cat(torch.split(KL, K, dim=-1), dim=0)
    VL = torch.cat(torch.split(VL, K, dim=-1), dim=0)
    attention = torch.matmul(QL, KL.transpose(2, 3))  # [128, 1, 325, 8]*[128, 1, 8, 325]---------->[128, 1, 325, 325]
    attention /= (d ** 0.5)
    attention = F.softmax(attention, dim=-1)
    X = torch.matmul(attention, VL)  # [128, 1, 325, 325]*[128, 1, 325, 8]--------->X[128, 1, 325, 8]
    X = torch.cat(torch.split(X, batch_size, dim=0),
                  dim=-1)  # orginal K, change to batch_size         将X(128, 1, 325, 8)分成8块torch.Size([16, 1, 325, 8])再在最后一维拼起来得到X（[16, 1, 325, 64]）
    return X
