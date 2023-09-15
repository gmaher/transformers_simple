import torch
from torch.nn import functional as F
import numpy as np

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, block_size,
    activation=torch.nn.Identity()):
        super(MultiHeadAttention,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.activation = activation

        self.Wa = torch.nn.Linear(input_size, num_heads*3*hidden_size, bias=False)

        self.Wout = torch.nn.Linear(hidden_size*num_heads, output_size)

        self.mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(
        1,1,self.block_size,self.block_size
        )

    def forward(self, x):
        """
            x - (Nbatch, Length, Vec size)
        """
        nb, l, e = x.size()

        tmp = self.Wa(x)
        tmp = tmp.view(nb, l, self.num_heads, 3*self.hidden_size).transpose(1,2)
        #tmp is now (nb, num_heads, l, 3*hidden_size)

        q,k,v = torch.split(tmp, self.hidden_size, dim=3)

        # A is (nb, num_heads, l, l)
        A = q @ k.transpose(-2,-1)/np.sqrt(self.hidden_size)
        A = A.masked_fill(self.mask[:,:,:l,:l] ==0, float('-inf'))
        A = F.softmax(A,dim=-1)

        o = A @ v #(nb, num_heads, l, hidden_size)

        o = o.transpose(1,2).contiguous().view(nb,l,self.hidden_size*self.num_heads)
        o = self.Wout(o)

        o = self.activation(o)

        return o

class TransformerBlock(torch.nn.Module):
    def __init__(self):
        pass
