# pytorch_segmentation_models
using pytorch to implement different segmewntation models, including both 2D and 3D.

There are lots of public pytorch based segmentation models, but they are designed for 2D image. For medicine image, it is usually 3D. Thus, I want to implement pytorch based 3D segmentation models.

***
##UNET
I refer to a excellent segmentation [code](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/keras/segmentation_models), and implement myself 3D segmentation models.


	dimension=2 # for two dimensions
    dimension=3 # for three dimensions

```
from segmentation_models import Unet
import torch
model = Unet(dimension=dimension, channel_in=1, basefilter=32, classes=2, pretrained=False).cuda()
img = torch.rand(size=[1, 1, 128, 128], dtype=torch.float32).cuda()
p = model(img)
print(p.shape)
```


The `dimension` parameter was used to decide whether this model is for 3D or 2D.
when out of memory is reported, we may reduce `basefilter` or `decoder_filters`.
if `pretrained` is true, the encode part would load pretrained model. But it only work when `dimension=2`, `channel_in=1`, `basefilter=64`

*==**Please look into UNET for more explanations of parameters.**==*

Currently, **only** resnet/vgg backbone is implement.

***

##High resolution segmentation

This module is modified according to the [HRNET-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master). 

A demo is:

	from segmentation_models import Unet, HighResolutionNet
    import torch
    model = HighResolutionNet(dimension=3, channel_in=3, classes=2, configureType='HRNET18')
    img = torch.rand(size=[1, 3, 128, 128, 64], dtype=torch.float32)
    p = model(img)
    print(p.shape)
    
`dimension=3` indicates the input image should be a three dimensions image, while `dimension=2` indicate the input image should be a two dimensions image. There configure types are provided: `HRNET48`, `HRNET32`, `HRNET18`. Please have a look for the original paper for more explanations.

If the input image size is (X, Y, Z), the output is (X/4, Y/4, Z/4). In the original [code](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/77044e60b6d71489465531ea60ba209b63145bcc/lib/core/criterion.py#L22), the loss should be calculate as following:
	
    h, w = label.size(1), label.size(2)
	score = F.upsample(input=p, size=(h, w), mode='bilinear)
    loss = nn.CrossEntropyLoss()(score, label)
    
However, I think the `F.upsample` could only provide a coarse mask. Thus, I replace the original `last_layer` with a two layer `ConvTranspose`. If it would cost much GPU memory, I think we can replace the two `ConvTranspose` with `F.upsample` + `ConvTranspose`. 

	model = HighResolutionNet(dimension=3, channel_in=3, classes=2, configureType='HRNET18', upsample=True)

Please note: if we set the `activation` to `sigmoid`, we need to use `nn.BCELoss`.