import torch
import torch.nn as nn

def getOperation(dimension=2):

    assert (dimension==2 or dimension==3), 'wrong dimension'

    if dimension == 2:
        Conv = nn.Conv2d
        BatchNorm = nn.BatchNorm2d
        maxpool = nn.MaxPool2d
        ConvTranspose = nn.ConvTranspose2d
    elif dimension == 3:
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        maxpool = nn.MaxPool3d
        ConvTranspose = nn.ConvTranspose3d

    return Conv, BatchNorm, maxpool, ConvTranspose

def getOutputChannel(layer):
    channel = 0
    if layer._modules:
        for _, subLayer in layer._modules.items():
            if subLayer._modules: # 如果modules有东西，就去查找channel是多少
                channel = getOutputChannel(subLayer)
            elif isinstance(subLayer, nn.Conv2d) or isinstance(subLayer, nn.Conv3d):
                channel = subLayer.out_channels
    else:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv3d):
            channel = layer.out_channels
        else:
            channel = 0
    return channel

def getChannel(model, name):
    modules = model._modules
    channel = 0
    for layName, layer in modules.items():
        if getOutputChannel(layer) != 0:
            channel = getOutputChannel(layer)
        if name == layName:
            return channel