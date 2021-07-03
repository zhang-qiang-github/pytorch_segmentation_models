# pytorch_segmentation_models
using pytorch to implement different segmewntation models, including both 2D and 3D.

There are lots of public pytorch based segmentation models, but they are designed for 2D image. For medicine image, it is usually 3D. Thus, I want to implement pytorch based 3D segmentation models.

I refer to a excellent segmentation [code](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/keras/segmentation_models), and implement myself 3D segmentation models.

***
### How to use it

For 2D:

```
from segmentation_models import Unet
import torch
model = Unet(dimension=2, channel_in=1, basefilter=32, classes=2, pretrained=False).cuda()
img = torch.rand(size=[1, 1, 128, 128], dtype=torch.float32).cuda()
p = model(img)
print(p.shape)
```

For 3D:

```
from segmentation_models import Unet
import torch
model = Unet(dimension=3, channel_in=1, basefilter=32, classes=2, pretrained=False).cuda()
img = torch.rand(size=[1, 1, 128, 128, 64], dtype=torch.float32).cuda()
p = model(img)
print(p.shape)
```

The `dimension` parameter was used to decide whether this model is for 3D or 2D.
when out of memory is reported, we may reduce `basefilter` or `decoder_filters`.
if `pretrained` is true, the encode part would load pretrained model. But it only work when `dimension=2`, `channel_in=1`, `basefilter=64`

***
Currently, **only** resnet/vgg backbone is implement.