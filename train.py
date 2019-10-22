# @Time: 2019/9/30 9:47
# @Author: jwzheng
# @Function： 读取图片

from torch.autograd import Variable
from torch import nn
from torch import optim
import torch
import os
# 模型训练参数类
# from config.conf import conf




def train(net,split,train_loader,test_loader,opt):
    if opt.use_gpu:
        # 设置在第一块显卡上运行
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_id
    # 创建模型保存的路径和训练日志路径
    if os.path.exists(opt.model_save_path) == False:
        os.makedirs(opt.model_save_path)
    if os.path.exists(opt.log_save_path) == False:
        os.makedirs(opt.log_save_path)

    log_out = open(opt.log_save_path+'/training_'+opt.model_name+'.log','w',encoding='utf8')
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(),conf.learning_rate)
    optimizer = optim.Adam(net.parameters(),lr=opt.learning_rate,weight_decay=5e-4)
    # for param in net.parameters():
    #     print('参数是{0}，训练方式{1}'.format(param,param.requires_grad))
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),1e-3)
    # 开始训练
    losses =[]
    acces = []
    eval_losses = []
    eval_acces = []
    count = 0
    best_eval_acc = 0
    # gpu上运行需要下面两句代码，cpu则不需要
    if opt.use_gpu:
        device = torch.device('cuda')
        net.to(device)
    for epoch in range(0,opt.epochs):
        print('开始训练第{}个epoch'.format(epoch))
        train_loss = 0
        train_acc = 0
        net.train()
        for inputs,labels in train_loader:
            count += 1
            # if count%100==0:
            #     print('训练第{}次'.format(count))
            print('训练第{}次'.format(count))
            # 这个地方很重要 不转换跑不了
            if opt.use_gpu:
                # gpu上训练
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                # cpu上训练
                inputs = inputs.float()
                labels = labels.long()

            inputs = Variable(inputs)
            labels = Variable(labels)

            # inputs shape [batch_size,3,size,size]
            # step1: hog [batch_size,hog_vector]
            #  out = net(inputs,hog)

            # 这是固定的五个步骤
            # 前向传播
            out = net(inputs)
            # print('out is: \n',out)
            # print('out.shape is: ',out.shape)
            loss = criterion(out,labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录结果 基本也可以模仿下面的代码即可
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _,pred = out.max(1)
            num_correct = (pred == labels).sum().item()
            acc = num_correct / inputs.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

        eval_loss = 0
        eval_acc = 0
        net.eval()
        with torch.no_grad():
            # 测试集不训练
            for img , label in test_loader:
                if opt.use_gpu:
                    # gpu上测试
                    img = img.cuda()
                    label = label.cuda()
                else:
                    # cpu上测试
                    img = img.float()
                    label = label.long()

                img = Variable(img)
                label = Variable(label)

                out = net(img)
                loss = criterion(out,label)

                # 记录误差
                eval_loss += loss.item()
                # print('out is: \n',out)
                _ , pred = out.max(1)
                num_correct = (pred==label).sum().item()
                acc = num_correct / img.shape[0]
                eval_acc += acc
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))

        print('Split {} Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(split,
            epoch+1, train_loss / len(train_loader),train_acc / len(train_loader), eval_loss / len(test_loader), eval_acc / len(test_loader)))

        # 写入日志
        log_out.write('Split {} Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(split,
            epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader),
            eval_loss / len(test_loader), eval_acc / len(test_loader))+'\n')
        if eval_acc / len(test_loader)> best_eval_acc:
            torch.save(net.state_dict(), opt.model_save_path+'/model_{}_split{}_ep{}.pth'.format(opt.model_name,split,epoch + 1))
            best_eval_acc = eval_acc / len(test_loader)
    print('训练结束\n')
    print('eval_losses is: ',eval_losses)
    print('eval_acces is: ',eval_acces)