# @Time: 2019/9/30 15:07
# @Author: jwzheng
# @Function：自定义的se_resnext
# pytorch的预训练模型中没有实现inceptionresnetv2，需要自己实现


from pretrainedmodels import inceptionresnetv2
from torch import nn
# from config.conf import conf
import torch

def get_inception_resnet_v2_pretrained_model(num_classes):
    pretrained_inception_resnet_v2 = inceptionresnetv2()
    pretrained_inception_resnet_v2.avgpool_1a = nn.AdaptiveAvgPool2d((1,1))
    pretrained_inception_resnet_v2.last_linear = nn.Linear(1536,num_classes)
    return pretrained_inception_resnet_v2

if __name__ == '__main__':
    pretrained_inception_resnet_v2 = get_inception_resnet_v2_pretrained_model()
    # print('pretrained_inceptionresnetv2 is:\n',pretrained_inception_resnet_v2)
    # test
    from torchsummaryX import summary
    summary(pretrained_inception_resnet_v2, torch.zeros((1, 3, 224, 224)))
    # for name, module in pretrained_inception_resnet_v2.named_modules():
    #     print('name is: ',name,'module is: ',module)

    # x = torch.randn((1, 3, 224, 224))
    # # print('len(pretrained_inception_resnet_v2.named_modules()) is: ',len(pretrained_inception_resnet_v2.named_modules()))
    # for name, module in pretrained_inception_resnet_v2.named_modules():
    #     # pretrained_inception_resnet_v2.modules()
    #     print('name is: ',name)
    #     print('module is: ',module)
    #     # input_size = x.shape
    #     # x = module(x)
    #     # output_size = x.shape
    #     # print("Layer: {} input size: {} output size: {}".format(name, input_size, output_size))