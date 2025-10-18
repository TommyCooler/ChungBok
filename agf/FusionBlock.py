import torch
import torch.nn as nn


#Concatenation
class ConcatFusion(nn.Module):
    def __init__(self, input_channels1, input_channels2,input_channels3):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=input_channels1 + input_channels2+input_channels3,
            out_channels=input_channels1,
            kernel_size=1
        )

    def forward(self, x1, x2,x3):
        # x1: [B, C1, T], x2: [B, C2, T]
        x = torch.cat((x1, x2,x3), dim=1)  # [B, C1 + C2, T]
        out = self.proj(x)             # [B, C1, T]
        return out

#Addition
class AddFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x1,x2,x3):
        return x1+x2+x3


class AvgFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x1,x2,x3):
        return( x1+x2+x3)/3

class MulFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x1,x2,x3):
        return x1*x2*x3

class TripConFusion(nn.Module):
    def __init__(self, input_channels1, input_channels2,input_channels3):
        super().__init__()
        self.proj1 = nn.Conv1d(input_channels1, input_channels1, kernel_size=1)
        self.proj2 = nn.Conv1d(input_channels2, input_channels1, kernel_size=1)
        self.proj3 = nn.Conv1d(input_channels3, input_channels1, kernel_size=1)

    def forward(self, x1, x2,x3):
        # x1: [B, C, T], x2: [B, C2, T]
        h1 = self.proj1(x1)  # → [B, C, T]
        h2 = self.proj2(x2)  # → [B, C, T]
        h3 = self.proj3(x3)  # → [B, C, T]
        return h1 * h2 *h3       # element-wise fusion, shape: [B, C, T]
