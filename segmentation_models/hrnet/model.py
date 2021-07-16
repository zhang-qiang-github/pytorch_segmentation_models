import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import getOperation
from ..backbones import Bottleneck, BasicBlock
from .blocks import HighResolutionModule
import numpy as np

ALIGN_CORNERS = None

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
configure = {
    'HRNET48': {
        'STAGE1':{'NUM_CHANNELS':64, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':4},
        'STAGE2':{'NUM_CHANNELS':[48, 96], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'},
        'STAGE3':{'NUM_CHANNELS':[48, 96, 192], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4, 4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'},
        'STAGE4':{'NUM_CHANNELS':[48, 96, 192, 384], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4, 4, 4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'},
    },

    'HRNET32': {
        'STAGE1':{'NUM_CHANNELS':64, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':4},
        'STAGE2':{'NUM_CHANNELS':[32, 64], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'},
        'STAGE3':{'NUM_CHANNELS':[32, 64, 128], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4, 4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'},
        'STAGE4':{'NUM_CHANNELS':[32, 64, 128, 256], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4, 4, 4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'},
    },

    'HRNET18': {
        'STAGE1':{'NUM_CHANNELS':64, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':4},
        'STAGE2':{'NUM_CHANNELS':[18, 36], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'},
        'STAGE3':{'NUM_CHANNELS':[18, 36, 72], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4, 4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'},
        'STAGE4':{'NUM_CHANNELS':[18, 36, 72, 144], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4, 4, 4, 4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'},
    },
}

class HighResolutionNet(nn.Module):

    def __init__(self,
                 channel_in=1,
                 configureType='HRNET18',
                 classes=1,
                 activation=None,
                 dimension=2,
                 upsample=False
                 ):
        super().__init__()
        self.activationFunc = activation
        self.upsample = upsample
        Conv, BatchNorm, maxpool, ConvTranspose = getOperation(dimension=dimension)

        # stem net
        self.conv1 = Conv(channel_in, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)

        stage1_cfg = configure[configureType]['STAGE1']
        num_channels = stage1_cfg['NUM_CHANNELS']
        block = blocks_dict[stage1_cfg['BLOCK']]
        num_blocks = stage1_cfg['NUM_BLOCKS']
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks, dimension=dimension)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = configure[configureType]['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels, dimension=dimension)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels, dimension=dimension)

        self.stage3_cfg = configure[configureType]['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels, dimension=dimension)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels, dimension=dimension)

        self.stage4_cfg = configure[configureType]['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels, dimension=dimension)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True, dimension=dimension)

        last_inp_channels = np.int(np.sum(pre_stage_channels))
        self.last_layer = nn.Sequential(
            Conv(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm(last_inp_channels),
            nn.ReLU(inplace=True),
            Conv(
                in_channels=last_inp_channels,
                out_channels=classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        self.upLayer = nn.Sequential(
            ConvTranspose(
                in_channels=last_inp_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            BatchNorm(64),
            nn.ReLU(inplace=True),
            ConvTranspose(
                in_channels=64,
                out_channels=classes,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dimension=2):
        Conv, BatchNorm, maxpool, ConvTranspose = getOperation(dimension=dimension)
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dimension=dimension))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, dimension=dimension))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer, dimension=2):
        Conv, BatchNorm, maxpool, ConvTranspose = getOperation(dimension=dimension)
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        Conv(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm(
                            num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        Conv(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True, dimension=2):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output,
                                      dimension=dimension
                                     )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        if len(x[0].shape) == 5: # three dimensions
            x0_h, x0_w, x0_k = x[0].size(2), x[0].size(3), x[0].size(4)
            x1 = F.interpolate(x[1], size=(x0_h, x0_w, x0_k), mode='trilinear', align_corners=ALIGN_CORNERS)
            x2 = F.interpolate(x[2], size=(x0_h, x0_w, x0_k), mode='trilinear', align_corners=ALIGN_CORNERS)
            x3 = F.interpolate(x[3], size=(x0_h, x0_w, x0_k), mode='trilinear', align_corners=ALIGN_CORNERS)
        elif len(x[0].shape) == 4: # two dimensions
            x0_h, x0_w = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)

        if self.upsample == False:
            x = self.last_layer(x)
        else:
            x = self.upLayer(x)

        if self.activationFunc == 'sigmoid':
            x = nn.Sigmoid()(x)
        elif self.activationFunc == 'relu':
            x = nn.ReLU()(x)
        elif self.activationFunc == 'softmax':
            x = nn.Softmax()(x)
        elif self.activationFunc == None:
            pass
        return x