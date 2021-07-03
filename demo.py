from segmentation_models import Unet
import torch
model = Unet(dimension=2, channel_in=3, basefilter=32, classes=2, backbone_name='resnet50', pretrained=True).cuda()
img = torch.rand(size=[1, 3, 128, 128], dtype=torch.float32).cuda()
p = model(img)
print(p.shape)

# model = Unet(dimension=3, channel_in=1, basefilter=32, classes=2, backbone_name='resnet50', pretrained=False).cuda()
# img = torch.rand(size=[1, 1, 128, 128, 64], dtype=torch.float32).cuda()
# p = model(img)
# print(p.shape)