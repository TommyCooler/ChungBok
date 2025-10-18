import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math

class CustomLinear(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple):  # (input: channels, length)  ; output: (i_channels, length)
        super().__init__()
        # print(output_shape, input_shape)
        self.i_shape = input_shape
        self.o_shape = output_shape

        if (output_shape[1] == input_shape[1]):
            self.weights1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[0]))
            self.bias1 = nn.Parameter(torch.Tensor(output_shape[1], output_shape[1]))
            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights1, mean=0.0, std=0.001)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)  # bias init

        if (output_shape[1] != input_shape[1]):
            self.weights1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[0]))
            self.bias1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[1]))
            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights1, mean=0.0, std=0.001)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)  # bias init

            self.weights2 = nn.Parameter(torch.Tensor(input_shape[1], output_shape[1]))
            self.bias2 = nn.Parameter(torch.Tensor(output_shape[0], output_shape[1]))

            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights2, mean=0.0, std=0.001)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias2, -bound, bound)  # bias init



    def forward(self, x):
        if (self.o_shape[1] != self.i_shape[1]):
            x = torch.add(torch.matmul(self.weights1, x ), self.bias1)
            x = torch.add(torch.matmul(x, self.weights2 ), self.bias2)
            return x

        x = torch.add(torch.matmul(self.weights1, x ), self.bias1)
        return x


class CustomNonLinear(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple, activation):  # (input: channels, length)  ; output: (i_channels, length)
        super().__init__()
        # print(output_shape, input_shape)
        self.i_shape = input_shape
        self.o_shape = output_shape
        if activation =='relu':
            self.act = nn.ReLU()
        elif activation =='gelu':
            self.act = nn.GELU()
        elif activation =='silu':
            self.act = nn.SiLU()
        elif activation =='elu':
            self.act = nn.ELU()
        elif activation =='leak_relu':
            self.act = nn.LeakyReLU()
        elif activation =='swish':
            self.act = nn.Hardswish()

        if (output_shape[1] == input_shape[1]):
            self.weights1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[0]))
            self.bias1 = nn.Parameter(torch.Tensor(output_shape[1], output_shape[1]))
            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights1, mean=0.0, std=0.01)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)  # bias init

        if (output_shape[1] != input_shape[1]):
            self.weights1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[0]))
            self.bias1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[1]))
            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights1, mean=0.0, std=0.01)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)  # bias init

            self.weights2 = nn.Parameter(torch.Tensor(input_shape[1], output_shape[1]))
            self.bias2 = nn.Parameter(torch.Tensor(output_shape[0], output_shape[1]))

            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights2, mean=0.0, std=0.01)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias2, -bound, bound)  # bias init



    def forward(self, x):
        if (self.o_shape[1] != self.i_shape[1]):
            x = self.act(torch.add(torch.matmul(self.weights1, x ), self.bias1))
            x = self.act(torch.add(torch.matmul(x, self.weights2 ), self.bias2))
            return x

        x = self.act(torch.add(torch.matmul(self.weights1, x ), self.bias1))
        return x

if __name__ == '__main__':
    pass

    x2 = torch.rand(size=(64,26, 15)).cuda()  # 26:number of channel, 15 time-length
    print(x2.shape)

    n3=CustomLinear(input_shape=(64,26,15),output_shape=(64,5,1), isactivation=True).cuda()
    opp=n3(x2)
    print(opp.shape)