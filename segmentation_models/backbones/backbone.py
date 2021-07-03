
# from .classification_models.classification_models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
# from .classification_models.classification_models import ResNeXt50, ResNeXt101
#
# from .inception_resnet_v2 import InceptionResNetV2
# from .inception_v3 import InceptionV3
#
# from keras.applications import DenseNet121, DenseNet169, DenseNet201
# from keras.applications import VGG16
# from keras.applications import VGG19


from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg16, vgg19
from torchvision.models import inception, inception_v3
from torchvision.models import densenet161, densenet169, densenet121, densenet201
backbones = {
    "vgg16": vgg16,
    "vgg19": vgg19,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    # "resnext50": ResNeXt50,
    # "resnext101": ResNeXt101,
    # "inceptionresnetv2": InceptionResNetV2,

    # "inceptionv3": inception_v3,
    # "densenet121": densenet121,
    # "densenet169": densenet169,
    # "densenet201": densenet201,

}

def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)