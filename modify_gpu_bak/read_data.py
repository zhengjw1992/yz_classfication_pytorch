# @Time: 2019/9/30 9:47
# @Author: wanglj
# @Function： 读取图片
import os
# 导入torch的相关包
import torch
import torchvision
from torch.utils.data import TensorDataset
from torch import nn
from torch import optim

# 导入自定义的包和方法
from dataset import MyDataset

# dir2label_dict = {'yes':1,'no':0}
from modify_gpu_bak.set_seed import seed_torch

dir2label_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                  '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18,
                  '19': 19,
                  '20': 20, '21': 21, '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27,
                  '28': 28, '29': 29}
batch_size = 64
model_name = 'resnext50'
num_classes = len(dir2label_dict)


def get_pretrain_model(model_name):
    if model_name == 'resnext50':
        return get_resnext50_pretrained_model()
    if model_name == 'resnext101':
        return get_resnext101_pretrained_model()
    if model_name == 'se_resnext101':
        return get_se_resnext101_pretrained_model()


def get_resnet18_pretrained_model():
    print('使用resnet18网络进行模型训练')
    from torchvision.models import resnet18
    pretrain_resnet = resnet18(pretrained=True)
    pretrain_resnet.fc = nn.Linear(512, num_classes)
    return pretrain_resnet


def get_resnet34_pretrained_model():
    print('使用resnet34网络进行模型训练')
    from torchvision.models import resnet34
    pretrain_resnet = resnet34(pretrained=True)
    pretrain_resnet.fc = nn.Linear(512, num_classes)
    return pretrain_resnet


def get_resnext50_pretrained_model():
    print('使用resnext50网络进行模型训练')
    from torchvision.models import resnext50_32x4d
    pretrained_resnext50 = resnext50_32x4d(pretrained=True)
    pretrained_resnext50.fc = nn.Linear(2048, num_classes)
    return pretrained_resnext50


def get_resnext101_pretrained_model():
    print('使用resnext101网络进行模型训练')
    from torchvision.models import resnext101_32x8d
    pretrained_resnext50 = resnext101_32x8d(pretrained=True)
    pretrained_resnext50.fc = nn.Linear(2048, num_classes)
    return pretrained_resnext50


def get_se_resnext101_pretrained_model():
    print('使用se_resnext101网络进行模型训练')
    from model import se_resnext
    pretrained_seresnext101 = se_resnext.get_se_resnext101_pretrained_model()
    return pretrained_seresnext101


def train(train_loader, test_loader):
    log_out = open('training.log','w',encoding='utf8')
    net = get_pretrain_model(model_name=model_name).cuda()
    # print('使用的模型结构是：\n',net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 1e-3, momentum=0.9, nesterov=True, weight_decay=5e-4)
    best_score = 0.
    epochs = 20
    # 开始训练
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    count = 0
    best_eval_acc = 0
    for epoch in range(0, epochs):
        print('开始训练第{}个epoch'.format(epoch))
        train_loss = 0
        train_acc = 0
        net.train()
        for inputs, labels in train_loader:
            count += 1
            # print('shape is: ',inputs.shape,labels.shape)
            print('训练第{}次'.format(count))
            # 这个地方很重要 不转换跑不了
            inputs = inputs.cuda()
            labels = labels.cuda()

            # 这是固定的五个步骤
            # 前向传播
            out = net(inputs)
            # print('out is: \n',out)
            # print('out.shape is: ',out.shape)
            loss = criterion(out, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录结果 基本也可以模仿下面的代码即可
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == labels).float().sum().item()
            acc = num_correct / inputs.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

        eval_loss = 0
        eval_acc = 0
        net.eval()
        with torch.no_grad():
            # 测试集不训练
            for img, label in test_loader:
                img = img.cuda()
                label = label.cuda()

                out = net(img)
                loss = criterion(out, label)

                # 记录误差
                eval_loss += loss.item()
                # print('out is: \n',out)
                _, pred = out.max(1)
                num_correct = (pred == label).float().sum().item()
                acc = num_correct / img.shape[0]
                eval_acc += acc

        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))


        print('Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(
            epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader),
            eval_loss / len(test_loader), eval_acc / len(test_loader)))
        # 写入日志
        log_out.write('Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(
            epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader),
            eval_loss / len(test_loader), eval_acc / len(test_loader))+'\n')
        if eval_acc / len(test_loader)> best_eval_acc:
            torch.save(net.state_dict(), 'model_ep{}.pth'.format(epoch + 1))
            best_eval_acc = eval_acc / len(test_loader)




# 读取数据集 生成满足Dataset格式的
def read_data(directory):
    filename_list = []
    label_list = []
    sub_directory_list = os.listdir(directory)  # train_data or validation_data
    # 可以考虑将sub_directory_list写死 以指定读取的label顺序
    for sub_directory in sub_directory_list:
        sub_directory_path = os.path.join(directory, sub_directory)  # 每一个子类别
        image_list = os.listdir(sub_directory_path)
        for image in image_list:
            image_path = os.path.join(sub_directory_path, image)
            filename_list.append(image_path)
            label_list.append(dir2label_dict[sub_directory])
    return filename_list, label_list


def data_augmentation():
    # augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
    #                                                      torchvision.transforms.RandomCrop(224),
    #                                                      torchvision.transforms.RandomHorizontalFlip(),
    #                                                      torchvision.transforms.ToTensor(),
    #                                                      torchvision.transforms.Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])
    #                                                      ])
    augmentation = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return augmentation


if __name__ == '__main__':
    seed_torch(1234)
    # 读取训练集
    # train_filename_list,train_label_list = read_data('mnist_pytorch/test/train_data')
    train_filename_list, train_label_list = read_data('data/train_image')
    augmentation = data_augmentation()
    train_dataset = MyDataset(filenames=train_filename_list, labels=train_label_list,
                              transform=augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               pin_memory=True)

    # test_filename_list,test_label_list = read_data('mnist_pytorch/test/validation_data')
    test_filename_list, test_label_list = read_data('data/validation_image')
    test_dataset = MyDataset(filenames=test_filename_list, labels=test_label_list,
                             transform=augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              pin_memory=True)

    train(train_loader, test_loader)
