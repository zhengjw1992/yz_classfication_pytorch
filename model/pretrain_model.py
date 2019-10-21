# @Time: 2019/10/8 11:27
# @Author: jwzheng
# @Function：
# from config.conf import conf
from torch import nn


def get_pretrain_model(model_name,num_classes):
    if model_name == 'resnet18':
        return get_resnet18_pretrained_model(num_classes)
    if model_name == 'resnet34':
        return get_resnet34_pretrained_model(num_classes)
    if model_name == 'resnext50':
        return get_resnext50_pretrained_model(num_classes)
    if model_name == 'resnext101':
        return get_resnext101_pretrained_model(num_classes)
    if model_name == 'se_resnext101':
        return get_se_resnext101_pretrained_model(num_classes)
    if model_name == 'inception_resnet_v2':
        return get_inception_resnet_v2_pretrained_model(num_classes)
    if model_name == 'senet154':
        return get_senet154_pretrained_model(num_classes)



def get_resnet18_pretrained_model(num_classes):
    print('加载resnet18模型 ')
    from torchvision.models import resnet18
    pretrain_resnet = resnet18(pretrained=True)
    pretrain_resnet.fc = nn.Linear(512,num_classes)
    return pretrain_resnet


def get_resnet34_pretrained_model(num_classes):
    print('加载resnet34模型')
    from torchvision.models import resnet34
    pretrain_resnet = resnet34(pretrained=True)
    pretrain_resnet.fc = nn.Linear(512,num_classes)
    return pretrain_resnet


def get_resnext50_pretrained_model(num_classes):
    print('加载resnext50模型')
    from torchvision.models import resnext50_32x4d
    pretrained_resnext50 = resnext50_32x4d(pretrained=True)
    pretrained_resnext50.fc = nn.Linear(2048,num_classes)
    return pretrained_resnext50


def get_resnext101_pretrained_model(num_classes):
    print('加载resnext101模型')
    from torchvision.models import resnext101_32x8d
    pretrained_resnext50 = resnext101_32x8d(pretrained=True)
    pretrained_resnext50.fc = nn.Linear(2048,num_classes)
    return pretrained_resnext50


def get_se_resnext101_pretrained_model(num_classes):
    print('加载se_resnext101模型')
    from model import se_resnext
    pretrained_seresnext101 = se_resnext.get_se_resnext101_pretrained_model(num_classes)
    return pretrained_seresnext101


def get_inception_resnet_v2_pretrained_model(num_classes):
    print('加载inception_resnet_v2模型')
    from model import inception_resnet_v2
    pretrained_inception_resnet_v2 = inception_resnet_v2.get_inception_resnet_v2_pretrained_model(num_classes)
    return pretrained_inception_resnet_v2


def get_senet154_pretrained_model(num_classes):
    print('加载senet154模型')
    from model import senet
    pretrained_senet154 = senet.get_senet154_pretrained_model(num_classes)
    return pretrained_senet154


