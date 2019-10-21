# @Time: 2019/10/3 21:01
# @Author: jwzheng
# @Function：加载模型 预测
import torch
from model.pretrain_model import get_pretrain_model
import os
import dataset
from data_preprocess import data_augmentation
from torch.utils.data import TensorDataset, DataLoader, Dataset
# from config.conf import conf

import argparse


#  加载模型
def load_model(moedl_name,model_path,num_classes):
    model = get_pretrain_model(moedl_name,num_classes)
    model.load_state_dict(torch.load(model_path))
    print(model)
    return model


def read_filename_list(directory):
    image_list = os.listdir(directory)
    # data_out = open('out.csv','w',encoding='utf8')
    # count = 0
    filename_list = []
    label_list = []
    for image in image_list:
        # count +=1
        # if count %100==0:
        #     print('正在处理第{0}张图片'.format(count))
        image_path = os.path.join(directory,image)
        filename_list.append(image_path)
        # label_list.append(image.strip().split('.')[0])
        label_list.append(image)
    return filename_list,label_list


def predict(net,opt):
    if opt.use_gpu:
        device = torch.device('cuda')
        net.to(device)
    filename_list,label_list = read_filename_list(opt.directory)
    augmentation = data_augmentation(opt.img_resize,opt.img_random_crop)
    test_dataset = dataset.MyDataset(filename_list, label_list, augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=4, shuffle=False,
                                               pin_memory=True)
    data_out = open('out.csv','w',encoding='utf8')
    count = 0
    net.eval()
    with torch.no_grad():
        for input, label in test_loader:
            count+=1
            if count%100==0:
                print(count)
            if opt.use_gpu:
                # gpu上预测
                input = input.cuda()
            else:
                # cpu上预测
                input = input.float()
            out = net(input)
            _, pred = out.max(1)
            save_res(data_out,label,pred)
            # break
    data_out.close()


def save_res(data_out,filename,pred):
    for i in range(0,len(filename)):
        data_out.write(filename[i]+','+str(pred.numpy()[i]+1)+'\n')


if __name__ == '__main__':
    # directory = 'E:\收藏数据集\观云识天比赛\Test'
    # model_name = 'resnext101'
    # model_path = 'C:\Software\Functions\QQDocuments\\738760187\FileRecv\\model_ep7.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default='senet154',help='name of deep learning net')
    parser.add_argument("--model_path",type=str,default='output/model_save_dir/model_ep7.pth',help='path of trained net weights')
    parser.add_argument("--test_directory",type=str,default='data/Test',help='path of testing data')
    parser.add_argument("--dir2label_dict",type=dict,default={'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,'17':16,'18':17,'19':18,
                                                              '20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,'29':28},help='network used during the training process')
    parser.add_argument("--use_gpu", type=bool, default=True, help="weather to use gpu")
    opt = parser.parse_args()
    parser.add_argument("--num_classes", type=int, default=len(opt.dir2label_dict), help="number of class")
    net = load_model(opt.model_name,opt.model_path,opt.num_classes)
    predict(net,opt)
