import torch
from torch import nn
import numpy as np
import EEGNet

inputs = torch.rand(1,1,30,1000)
model = EEGNet.Input_conv(in_channels=1, out_channels=32)
output = model(inputs, inputs)
print(output.shape)

def test1():
    inputs = torch.rand(1,1,30,1000)
    ks1 = 3 # must be odd number
    ks2 = 25
    pad1 = int((ks1 - 1) / 2)
    pad2 = int((ks2 - 1) / 2)
    mod1 = nn.Conv2d(1,32,kernel_size=(ks1,ks2),padding=(pad1, pad2)) #torch.Size([1, 32, 5, 143])
    #mod1 = nn.Conv2d(1,32,kernel_size=ks1,padding=pad1) #torch.Size([1, 32, 5, 143])
    mod2 = nn.MaxPool2d((1,100))
    out = mod1(inputs)
    print(out.shape)
    out = mod2(out)
    print(out.shape)
    print(out[0,0,0:2,:])
    mod3 = nn.Flatten(2,-1)
    out = mod3(out)
    print(out.shape)
    out1 = torch.unsqueeze(out, -1)
    print(out1.shape)
    out2 = torch.unsqueeze(out, -2)
    print(out2.shape)
    ans = torch.matmul(out1, out2)
    print(ans.shape)
    print(out[0,0,0:20])


    #tensor1 = torch.tensor(np.array([[[1], [2], [3]]]))
    tensor1 = torch.rand(1,32,300,1)
    print(tensor1.size())
    #tensor2 = torch.tensor(np.array([[[4, 5, 6]]]))
    tensor2 = torch.rand(1,32,1,300)
    print(tensor2.size())
    ans = torch.matmul(tensor1, tensor2)
    print(ans.size())
    #print(ans)


    x = torch.tensor([1, 2, 3, 4])
    sque_a = torch.unsqueeze(x, 0)
    print(sque_a.size())
    sque_b = torch.unsqueeze(x, 1)
    print(sque_b.size())