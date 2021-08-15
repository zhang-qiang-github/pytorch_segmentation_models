import torch
img3D = torch.rand(size=[1, 1, 128, 128, 128], dtype=torch.float32).cuda()
img2D = torch.rand(size=[1, 1, 128, 128], dtype=torch.float32).cuda()

############# unet #######################
from segmentation_models import Unet
model = Unet(dimension=3, channel_in=1, backbone_name='vgg19', basefilter=64, classes=2, pretrained=False).cuda()
# model = Unet(dimension=3, channel_in=1, backbone_name='resnet50', basefilter=32, classes=2, pretrained=False).cuda()
p = model(img3D)
print(p.shape)

model = Unet(dimension=2, channel_in=1, backbone_name='vgg19', basefilter=64, classes=2, pretrained=False).cuda()
p = model(img2D)
print(p.shape)

############## HighResolutionNet ###################
from segmentation_models import HighResolutionNet
model = HighResolutionNet(dimension=3, channel_in=1, classes=2, configureType='HRNET18').cuda()
p = model(img3D)
print(p.shape)