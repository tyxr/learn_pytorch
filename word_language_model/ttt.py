a = range(0,10,2)
print(list(a))
print(list(enumerate(a)))
import torch
import torch.nn as nn

b = torch.rand((3,2,4), out=None)
print(b)
print(b.view(6,4))


a = torch.randn(5)

a.add_(3)
