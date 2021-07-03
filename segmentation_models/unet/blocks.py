
import torch
import torch.nn as nn
from ..utils import getOperation

def ConvRelu(channel_in, channel_out, kernel_size, use_batchnorm=False, dimension=2):
    Conv, BatchNorm, maxpool, ConvTranspose = getOperation(dimension=dimension)

    if use_batchnorm:
        layer = nn.Sequential(
            Conv(channel_in, channel_out, kernel_size=kernel_size, bias=False, padding=int((kernel_size-1)/2)),
            BatchNorm(channel_out),
            nn.ReLU()
        )
    else:
        layer = nn.Sequential(
            Conv(channel_in, channel_out, kernel_size=kernel_size, bias=True, padding=int((kernel_size-1)/2)),
            nn.ReLU()
        )
    return layer

class Upsample_block(nn.Module):

    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 upsample_rate=2,
                 use_batchnorm=False,
                 skipChannel=0,
                 dimension=2
                 ):
        super().__init__()
        if dimension == 3:
            mode='trilinear'
        elif dimension == 2:
            mode='bilinear'
        self.upsample = nn.Upsample(scale_factor=upsample_rate, mode=mode, align_corners=True)
        self.skipChannel = skipChannel
        channel_in += skipChannel
        self.layer1 = ConvRelu(channel_in, channel_out, kernel_size=kernel_size, use_batchnorm=use_batchnorm, dimension=dimension)
        self.layer2 = ConvRelu(channel_out, channel_out, kernel_size=kernel_size, use_batchnorm=use_batchnorm, dimension=dimension)

    def forward(self, x, skipFeature=None):

        x = self.upsample(x)
        if self.skipChannel != 0:
            if skipFeature is None:
                raise RuntimeError("should input a skip feature")
            x = torch.cat([x, skipFeature], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Transpose_block(nn.Module):

    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 transpose_kernel_size=4, # kernel一定要是奇数
                 upsample_rate=2,
                 use_batchnorm=False,
                 skipChannel=0,
                 dimension=2
                 ):
        super().__init__()
        Conv, BatchNorm, maxpool, ConvTranspose = getOperation(dimension=dimension)
        self.transposeConv = ConvTranspose(channel_in, channel_out, kernel_size=transpose_kernel_size, stride=upsample_rate, bias=not(use_batchnorm), padding=1)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = BatchNorm(channel_out)
        self.skipChannel = skipChannel
        channel_in += skipChannel
        self.layer = ConvRelu(channel_out+skipChannel, channel_out, kernel_size, use_batchnorm=use_batchnorm, dimension=dimension)

    def forward(self, x, skipFeature=None):

        x = self.transposeConv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = nn.ReLU(inplace=True)(x)
        if self.skipChannel != 0:
            if skipFeature is None:
                raise RuntimeError("should input a skip feature")
            x = torch.cat([x, skipFeature], dim=1)
        x = self.layer(x)
        return x