import torch
x = torch.rand(4, 96, 16, 16)
x = torch.split(x, int(96/12), 1)
print(len(x), x[0].shape)
