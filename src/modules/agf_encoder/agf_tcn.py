import torch
import torch.nn as nn
from sympy.codegen.fnodes import intent_out
from torch.nn.utils import weight_norm
from basicBlock import CustomLinear
from FusionBlock import MulFusion, AvgFusion, ConcatFusion, AddFusion, \
    TripConFusion

from Journals.JMLR2025.caf_tcn.modules.iat_net import IATNet


class SEBlock(nn.Module):
    def __init__(self, channels, activation, reduction=16):
        super().__init__()
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'elu': nn.ELU(),
            'leak_relu': nn.LeakyReLU(),
            'swish': nn.Hardswish()
        }
        self.act = activations.get(activation.lower(), nn.ReLU())

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            self.act,
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # print(x[0][0])
        # print(x[:, :, :-self.chomp_size].contiguous()[0][0])
        # print('-------------------')
        return x[:, :, :-self.chomp_size].contiguous()


class AttentionFusionBlock(nn.Module):
    def __init__(self, n_inputs,
                 n_outputs,
                 dilation,
                 activation,
                 fuse_type,
                 dropout=0.2):
        super(AttentionFusionBlock, self).__init__()

        # Activation
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'elu': nn.ELU(),
            'leak_relu': nn.LeakyReLU(),
            'swish': nn.Hardswish()
        }
        self.act = activations.get(activation.lower(), nn.ReLU())

        # Conv blocks (kernel size 3)
        self.conv11 = weight_norm(nn.Conv1d(n_inputs, n_outputs, 3, padding=2 * dilation, dilation=dilation))
        self.chomp11 = Chomp1d(2 * dilation)
        self.dropout11 = nn.Dropout(dropout)

        self.conv12 = weight_norm(nn.Conv1d(n_outputs, n_outputs, 3, padding=2 * dilation, dilation=dilation))
        self.chomp12 = Chomp1d(2 * dilation)
        self.dropout12 = nn.Dropout(dropout)

        self.se11 = SEBlock(n_outputs, activation)
        self.se12 = SEBlock(n_outputs, activation)

        # Conv blocks (kernel size 5)
        self.conv21 = weight_norm(nn.Conv1d(n_inputs, n_outputs, 5, padding=4 * dilation, dilation=dilation))
        self.chomp21 = Chomp1d(4 * dilation)
        self.dropout21 = nn.Dropout(dropout)

        self.conv22 = weight_norm(nn.Conv1d(n_outputs, n_outputs, 5, padding=4 * dilation, dilation=dilation))
        self.chomp22 = Chomp1d(4 * dilation)
        self.dropout22 = nn.Dropout(dropout)

        self.se21 = SEBlock(n_outputs, activation)
        self.se22 = SEBlock(n_outputs, activation)

        # Conv blocks (kernel size 7)
        self.conv31 = weight_norm(nn.Conv1d(n_inputs, n_outputs, 7, padding=6 * dilation, dilation=dilation))
        self.chomp31 = Chomp1d(6 * dilation)
        self.dropout31 = nn.Dropout(dropout)

        self.conv32 = weight_norm(nn.Conv1d(n_outputs, n_outputs, 7, padding=6 * dilation, dilation=dilation))
        self.chomp32 = Chomp1d(6 * dilation)
        self.dropout32 = nn.Dropout(dropout)
        self.se31 = SEBlock(n_outputs, activation)
        self.se32 = SEBlock(n_outputs, activation)

        # Fusion selector
        self.Fusion = self._select_fusion(fuse_type, n_outputs)

        # Initialize weights
        self.init_weights()

    def _select_fusion(self, fuse_type, n_outputs):
        if fuse_type == 1:
            return AddFusion()
        elif fuse_type == 2:
            return AvgFusion()
        elif fuse_type == 3:
            return MulFusion()
        elif fuse_type == 4:
            return ConcatFusion(input_channels1=n_outputs, input_channels2=n_outputs)
        elif fuse_type == 5:
            return TripConFusion(input_channels1=n_outputs,
                                 input_channels2=n_outputs,
                                 input_channels3=n_outputs)
        else:
            raise ValueError(f"Unsupported fuse_type: {fuse_type}")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0.0, 0.0001)
                # nn.init.normal_(m.weight, 0.0, 0.001)
                # nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # Branch 1
        x11 = self.act(self.chomp11(self.conv11(x)))
        x11 = self.dropout11(x11)
        x11 = self.se11(x11)

        x12 = self.act(self.chomp12(self.conv12(x11)))
        x12 = self.dropout12(x12)
        x12 = self.se12(x12)


        # Branch 2
        x21 = self.act(self.chomp21(self.conv21(x)))
        x21 = self.dropout21(x21)
        x21 = self.se21(x21)

        x22 = self.act(self.chomp22(self.conv22(x21)))
        x22 = self.dropout22(x22)
        x22 = self.se22(x22)


        # Branch 3
        x31 = self.act(self.chomp31(self.conv31(x)))
        x31 = self.dropout31(x31)
        x31 = self.se31(x31)

        x32 = self.act(self.chomp32(self.conv32(x31)))
        x32 = self.dropout32(x32)
        x32 = self.se32(x32)


        # Fusion
        out = self.Fusion(x12, x22, x32)
        return out


class Agf_TCN(nn.Module):
    def __init__(self, num_inputs,
                 num_channels,
                 dropout,
                 activation,
                 fuse_type,
                 window_size,
                #  mode,
                 ):
        super(Agf_TCN, self).__init__()
        self.layers = []
        # self.mode = mode
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            # dilation_size = 1
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            self.layers += [AttentionFusionBlock(n_inputs=in_channels, n_outputs=out_channels,
                                            dilation=dilation_size,
                                            activation=activation,
                                            dropout=dropout,
                                            fuse_type=fuse_type)]

        self.network = nn.Sequential(*self.layers)

        #reconstruction
        self.decoder = CustomLinear(input_shape=(num_channels[-1], window_size),
                                       output_shape=(num_inputs, 1))

        #contrastive learning
        # self.mlp = self._contrastive_mode_setting(mode,num_inputs)

    # def _contrastive_mode_setting(self, mode,dims):
    #     if mode =="train": 
    #         return nn.Linear(dims, 128)
    #     else: return None

    def forward(self, x, generated_labels):
        if self.mode=="train":
            x = self.network(x)   #encoder
            x = self.decoder(x)     #reconstruction
            return x

# if __name__ == '__main__':
#     pass
#     x = torch.rand(size=(2, 2, 16)).cuda()  # 5:batch_size, 1:number of channel, 26 time-length
#     print(x.shape)

#     labels = torch.rand(size=(32, 2, 1)).cuda()  # 5:batch_size, 1:number of channel, 26 time-length
#     print(labels.shape)

#     net = IATNet(n_inputs=2,length=16,num_channels=[32,32],dropout=0.2,
#                  activation='relu',fuse_type=1,tau=0.005,contrastive_mode=True,
#                  train_mode=False)
#     output = net(x,labels)
#     print(output.shape)
