import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,
     activation=nn.LeakyReLU(0.05), output_activation=nn.Identity()):
        super(FCN,self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.output_activation = output_activation

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size,hidden_size))
        for n in range(num_layers):
            self.layers.append(nn.Linear(hidden_size,hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self,x):
        o = x
        for l in self.layers[:-1]:
            o = l(o)
            o = self.activation(o)

        o = self.layers[-1](o)
        o = self.output_activation(o)

        return o
