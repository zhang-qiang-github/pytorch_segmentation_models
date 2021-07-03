
import torch
import torch.nn as nn
from ..backbones import get_backbone
from ..utils import getOperation, getChannel
from .blocks import Transpose_block, Upsample_block
DEFAULT_SKIP_CONNECTIONS = {
    'vgg16':            ('29', '22', '15', '8', '3'),
    'vgg19':            ('35', '26', '17', '8', '3'),
    'resnet18':         ('layer4', 'layer3', 'layer2', 'layer1', 'conv1'), # check 'bn_data'
    'resnet34':         ('layer4', 'layer3', 'layer2', 'layer1', 'conv1'),
    'resnet50':         ('layer4', 'layer3', 'layer2', 'layer1', 'conv1'),
    'resnet101':        ('layer4', 'layer3', 'layer2', 'layer1', 'conv1'),
    'resnet152':        ('layer4', 'layer3', 'layer2', 'layer1', 'conv1'),
    # 'resnext50':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    # 'resnext101':       ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    # 'inceptionv3':          (228, 86, 16, 9),
    # 'inceptionresnetv2':    (594, 260, 16, 9),
    # 'densenet121':          (311, 139, 51, 4),
    # 'densenet169':          (367, 139, 51, 4),
    # 'densenet201':          (479, 139, 51, 4),
}

class Unet(nn.Module):

    '''
    parameters explanation:
    channel_in: 3 for rgb, and 1 for gray image
    backbone_name: resnet, vgg, densenet and so on
    basefilter: many public models obtain 64 filter and then up/down sampling
                however, 64 filter usually cause out of memory. We can reduce the base filter.
    decoder_block_type: upsampling or transpose
    activation: sigmoid/relu/none, none is linear
    dimension: 3 for three dimensions medicine image, and 2 for two dimensions image
    pretrained: even pretrained=True, it would load pretrained model only when dimension=2 and channel_in=3
    '''
    def __init__(self,
                 channel_in=1,
                 backbone_name='resnet50',
                 basefilter=64,
                 decoder_block_type='upsampling',
                 decoder_filters=(256,128,64,32,16),
                 n_upsample_blocks=5,
                 upsample_rates=(2, 2, 2, 2, 2),
                 decoder_use_batchnorm=True,
                 classes=1,
                 activation='sigmoid',
                 dimension=2,
                 pretrained=False
                 ):
        super().__init__()
        self.n_upsample_blocks = n_upsample_blocks
        if decoder_block_type == 'upsampling':
            up_block = Upsample_block
        elif decoder_block_type == 'transpose':
            up_block = Transpose_block

        self.backbone = get_backbone(backbone_name, pretrained=pretrained, channel_in=channel_in, dimension=dimension, basefilter=basefilter)
        self.skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]
        Conv, BatchNorm, maxpool, ConvTranspose = getOperation(dimension=dimension)

        lastChannel = getChannel(self.backbone, self.skip_connections[0])
        for i in range(n_upsample_blocks):
            # upChannel = getChannel(self.backbone, self.skip_connections[i+1])
            if i+1 >= len(self.skip_connections):
                skipChannel = 0
            else:
                skipChannel = getChannel(self.backbone, self.skip_connections[i+1])
            upLayer = up_block(channel_in=lastChannel,
                               channel_out=decoder_filters[i],
                               skipChannel=skipChannel,
                               upsample_rate=upsample_rates[i],
                               use_batchnorm=decoder_use_batchnorm,
                               dimension=dimension
                               )
            lastChannel = decoder_filters[i]
            self.__setattr__('upLayer_'+str(i), upLayer)

        self.layer = Conv(in_channels=lastChannel, out_channels=classes, kernel_size=3, padding=1)
        self.activationFunc = activation

    def forward(self, x):

        backboneLayer = self.backbone._modules
        up_features = dict()
        for name, layer in backboneLayer.items():
            x = layer(x)
            if name in self.skip_connections:
                up_features[name] = x
            if name == self.skip_connections[0]:
                break

        for i in range(self.n_upsample_blocks):
            upLayer = self.__getattr__('upLayer_'+str(i))
            if i+1 >= len(self.skip_connections):
                skipFeature = None
            else:
                skipFeature = up_features[self.skip_connections[i+1]]
            x = upLayer(x, skipFeature=skipFeature)

        x = self.layer(x)
        if self.activationFunc=='sigmoid':
            x = nn.Sigmoid()(x)
        elif self.activationFunc=='relu':
            x = nn.ReLU()(x)
        elif self.activationFunc=='softmax':
            x = nn.Softmax()(x)
        elif self.activationFunc==None:
            pass
        return x
