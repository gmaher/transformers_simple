import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_length, input_channels, output_size,
     hidden_channels, kernel_size, num_layers,
     activation=nn.LeakyReLU(0.05),
     output_activation=nn.Identity()):
        super(CNN,self).__init__()

        self.input_length = input_length
        self.input_channels = input_channels
        self.output_size = output_size
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.activation = activation
        self.output_activation = output_activation

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv1d(input_channels, hidden_channels, kernel_size,
        padding=1))
        for n in range(num_layers):
            self.layers.append(nn.Conv1d(hidden_channels, hidden_channels,
             kernel_size, padding=1))

        self.output_layer_input_size = input_length*hidden_channels

        self.layers.append(nn.Linear(self.output_layer_input_size, output_size))

    def forward(self,x):
        o = x
        for l in self.layers[:-1]:
            o = l(o)
            o = self.activation(o)

        o = torch.flatten(o, start_dim=1)
        o = self.layers[-1](o)
        o = self.output_activation(o)

        return o
