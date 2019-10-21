# @Time: 2019/9/30 15:07
# @Author: wanglj
# @Function：自定义的se_resnext
# pytorch的预训练模型中没有实现se_resnext，需要自己实现


from pretrainedmodels import se_resnext101_32x4d
from torch import nn

def get_se_resnext101_pretrained_model():
    pretrained_seresnext101 = se_resnext101_32x4d()
    pretrained_seresnext101.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    pretrained_seresnext101.last_linear = nn.Linear(2048,2)
    return pretrained_seresnext101


if __name__ == '__main__':
    pretrained_seresnext101 = get_se_resnext101_pretrained_model()
    print(pretrained_seresnext101)