# @Time: 2019/10/11 9:28
# @Author: jwzheng
# @Function：

from pretrainedmodels import senet154
# from model.model_test import senet154
from torch import nn
# from config.conf import conf



def get_senet154_pretrained_model(num_classes):
    pretrained_senet154 = senet154()
    pretrained_senet154.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    pretrained_senet154.last_linear = nn.Linear(2048,num_classes)
    return pretrained_senet154


if __name__ == '__main__':
    pretrained_senet154 = get_senet154_pretrained_model()
    print(pretrained_senet154)
    # 查看网络结构
    # from torchsummaryX import summary
    # import torch
    # summary(pretrained_senet154, torch.zeros((1, 3, 224, 224)))
