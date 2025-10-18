import torch.nn as nn
from src.utils.padding import _pad_1d, _same_padding_lr
class TemporalConv1d(nn.Module):
    """
    Conv1d thời gian, giữ nguyên T:
      - causal=True  -> pad-trái (left = (k-1)*d), right=0
      - causal=False -> pad 'same' 2 phía (left/right theo _same_padding_lr)
    pad_mode: 'reflect' | 'replicate' | 'zeros'
    """
    def __init__(self, cin, cout, k: int, dilation: int = 1,
                 bias: bool = True, causal: bool = True, pad_mode: str = "reflect"):
        super().__init__()
        self.k = k
        self.d = dilation
        self.causal = causal
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(cin, cout, k, padding=0, dilation=dilation, bias=bias)

    def forward(self, x):  # x: (B, C_in, T)
        if self.causal:
            left = (self.k - 1) * self.d
            right = 0
        else:
            left, right = _same_padding_lr(self.k, self.d)

        x = _pad_1d(x, left, right, self.pad_mode)
        return self.conv(x)  # -> (B, C_out, T)
