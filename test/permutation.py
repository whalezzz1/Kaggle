import torch

"""permutation = torch.randperm(36458)
A = torch.rand(36458, 12, 325)
B = torch.rand((36458, 24, 2))
C = torch.rand(36458, 12, 325)
print("A:", A, '\n', B, '\n', C, '\n')
trainX = A[permutation]
trainTE = B[permutation]
trainY = C[permutation]
print(trainX, '\n', trainTE, '\n', trainY, '\n')"""
permutation = torch.randperm(10)
print("permutation:",permutation)
A = torch.rand(10, 3)
print("A:",A,A.shape)
A2 = A[permutation]
print("A2:",A2,A2.shape)

