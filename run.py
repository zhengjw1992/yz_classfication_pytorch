# @Time: 2019/10/8 10:38
# @Author: jwzheng
# @Function： 模型训练 和 预测 的入口
# 自定义包
import train
import data_preprocess
# from config.conf import conf
from dataset import MyDataset
from model.pretrain_model import get_pretrain_model

# 系统包
import torch
from torch.utils.data import DataLoader
import argparse
import datetime

# 入口方法
def run(opt):
    # 读取数据，读取成MyDataset类可以处理的格式
    train_filename_list,train_label_list = data_preprocess.read_data(train_directory=opt.train_directory,dir2label_dict=opt.dir2label_dict)
    # 定义一个数据增强的操作
    augmentation = data_preprocess.data_augmentation(opt.img_resize,opt.img_random_crop)
    # 使用MyDataset类和DataLoader类加载数据集
    train_dataset = MyDataset(filenames=train_filename_list, labels=train_label_list,transform=augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size, shuffle=True,
                                               pin_memory=True)

    # 同样的方式 加载验证数据集
    test_filename_list,test_label_list = data_preprocess.read_data(train_directory=opt.test_directory,dir2label_dict=opt.dir2label_dict)
    test_dataset = MyDataset(filenames=test_filename_list, labels=test_label_list,transform=augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=opt.batch_size, shuffle=True,
                                                pin_memory=True)

    # 定义一个网络
    net = get_pretrain_model(opt.model_name,opt.num_classes)

    # 训练集上训练、测试集上测试效果
    train.train(net,train_loader,test_loader,opt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir2label_dict",type=dict,default={'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,'17':16,'18':17,'19':18,
                                                              '20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,'29':28},help='network used during the training process')
    parser.add_argument("--model_name",type=str,default='senet154',help='network used during the training process')
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--train_directory", type=str, default='data/train_image', help="path of training data")
    parser.add_argument("--test_directory", type=str, default='data/validation_image', help="path of testing data")
    parser.add_argument("--use_gpu", type=bool, default=True, help="weather to use gpu")
    parser.add_argument("--cuda_id", type=str, default='0', help="which cuda used to run the code")
    parser.add_argument("--img_resize", type=int, default=399, help="size of each image dimension")
    parser.add_argument("--img_random_crop", type=int, default=336, help="size of each image dimension")
    opt = parser.parse_args()
    # print(opt)
    # print(opt.model_name)
    now_datetime = datetime.datetime.now()
    model_time = str(now_datetime)[0:10]+'_'+str(now_datetime)[11:19]
    parser.add_argument("--model_save_path", type=str, default='output/'+opt.model_name+'_'+model_time+'/model_save_dir', help="save path of training model")
    parser.add_argument("--log_save_path", type=str, default='output/'+opt.model_name+'_'+model_time+'/model_save_dir', help="save path of training log")
    parser.add_argument("--num_classes", type=int, default=len(opt.dir2label_dict), help="number of class")
    opt = parser.parse_args()
    print(opt)
    # print(opt.model_save_path)
    # print(opt.dir2label_dict)

    run(opt)