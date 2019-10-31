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
from sklearn.model_selection import StratifiedKFold
import tool


# 入口方法，可以进行交叉验证的训练
def run_cv(opt):
    # 读取读片，和之前不同的是，这里的训练集和验证集（测试集）在一个文件夹中，后面适用kfold随机划分训练集和验证集（测试集）
    filename_list,label_list = data_preprocess.read_data(directory=opt.train_directory,dir2label_dict=opt.dir2label_dict)
    # 分层抽样
    skfold = StratifiedKFold(n_splits=opt.cv_num,shuffle=False) # random_state=0 会使得每次run_cv()的训练集和测试集分割都一样
    for split, (train_index_list, val_index_list) in enumerate(skfold.split(label_list,label_list)):
        print('**********Split %d**********'%split)
        print('经过分层抽样后，训练集中的数据量为：{0}，验证集中的数据量为{1}。'.format (len(train_index_list),len(val_index_list)))
        train_label_num_dict = tool.count_class_num(label_list,train_index_list)
        val_label_num_dict = tool.count_class_num(label_list,val_index_list)

        train_label_num_dict=sorted(train_label_num_dict.items(),key=lambda x:x[0])
        val_label_num_dict=sorted(val_label_num_dict.items(),key=lambda x:x[0])
        print('训练集中各个类别的数据量为: ',train_label_num_dict)
        print('验证集中各个类别的数据量为: ',val_label_num_dict)

        # 定义数据增强操作
        augmentation = data_preprocess.data_augmentation(opt.img_resize,opt.img_random_crop,mode='train')

        # 根据分层抽样得到的数据index下标来获取训练集
        train_filename_list = tool.get_index_value(value_list=filename_list,index_list=train_index_list)
        train_label_list = tool.get_index_value(value_list=label_list,index_list=train_index_list)

        train_dataset = MyDataset(filenames=train_filename_list, labels=train_label_list, transform=augmentation)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=opt.batch_size, shuffle=True,
                                                 pin_memory=True)

        # 根据分层抽样得到的数据index下标来获取验证集
        val_filename_list = tool.get_index_value(value_list=filename_list,index_list=val_index_list)
        val_label_list = tool.get_index_value(value_list=label_list,index_list=val_index_list)

        val_dataset = MyDataset(filenames=val_filename_list, labels=val_label_list, transform=augmentation)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=opt.batch_size, shuffle=True,
                                                   pin_memory=True)

        # 定义一个网络
        net = get_pretrain_model(opt.model_name,opt.num_classes)

        # 训练集上训练、测试集上测试效果
        train.train(net,split,train_loader,val_loader,opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir2label_dict",type=dict,default={'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,'17':16,'18':17,'19':18,
                                                              '20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,'29':28},help='directory to label')

    # parser.add_argument("--dir2label_dict",type=dict,default={'有积雪':1,'无积雪':0},help='')
    parser.add_argument("--model_name",type=str,default='senet154',help='network used during the training process')
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--cv_num", type=int, default=5, help="cross validation")
    parser.add_argument("--train_directory", type=str, default='data/train_image', help="path of training data")
    parser.add_argument("--use_gpu", type=bool, default=True, help="weather to use gpu")
    parser.add_argument("--cuda_id", type=str, default='0', help="which cuda used to run the code")
    parser.add_argument("--img_resize", type=int, default=369, help="size of each image dimension") # 369
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
    run_cv(opt)