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
# tta
import ttach as tta
from ttach import aliases
from ttach.base import Compose


#  加载模型
def load_model(model_name,model_path,num_classes):
    model = get_pretrain_model(model_name,num_classes)
    model.load_state_dict(torch.load(model_path,map_location='cpu'))
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


def run(net,opt):
    if opt.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_id
        device = torch.device('cuda')
        net.to(device)
    filename_list,label_list = read_filename_list(opt.test_directory)
    # 使用测试时数据增强来修改下面这句代码
    augmentation = data_augmentation(opt.img_resize,opt.img_random_crop,mode='predict')

    test_dataset = dataset.MyDataset(filename_list, label_list, augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=opt.batch_size, shuffle=False,
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
            # 根据mode来选择是否需要使用tta来进行预测预测
            out = predict(net, input, mode=opt.predict_mode)
            _, pred = out.max(1)
            save_res(data_out,label,pred)
    data_out.close()


def predict(net, input, mode):
    if mode=='single':
        out = net(input)
        return out
    if mode =='pre_defined_tta':
        tta_net = tta.ClassificationTTAWrapper(net, aliases.d4_transform())
        out = tta_net.forward(input)
        return out
    if mode == 'defined_tta':
        out = defined_tta(net,input)
        return out


def save_res(data_out,filename,pred):
    for i in range(0,len(filename)):
        data_out.write(filename[i]+','+str(pred.numpy()[i]+1)+'\n')  # 不应该加1，应该去掉



# tta
def defined_tta(net,input):
    pass
# 下面的代码不要删掉，后面修改成自定义的tta
# def tta_test(net,image):
#     print('tta_test method...')
#     labels = []
#     masks = []
#     # defined 2 * 2 * 3 * 3 = 36 augmentations !
#     transforms = tta.Compose(
#         [
#             tta.HorizontalFlip(),
#             tta.Rotate90(angles=[0, 180]),
#             tta.Scale(scales=[1]),
#             tta.Multiply(factors=[0.9, 1, 1.1]),
#         ]
#     )
#     # augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(img_resize),
#     #                                                torchvision.transforms.RandomCrop(img_random_crop),
#     #                                                torchvision.transforms.RandomHorizontalFlip(),
#     #                                                torchvision.transforms.ToTensor(),
#     #                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
#     #                                                ])
#
#     for count,transformer in enumerate(transforms): # custom transforms or e.g. tta.aliases.d4_transform()
#         print('*******************count{}*****************'.format(count))
#         # augment image
#         augmented_image = transformer.augment_image(image)
#         print('原始图片:\n',type(image),image.shape)
#         print('tta后的图片:\n',type(augmented_image),augmented_image.shape)
#         # pass to model
#         model_output = net(augmented_image)
#         print('model_output\n',type(model_output),model_output.shape)
#         # reverse augmentation for mask and label
#         # deaug_mask = transformer.deaugment_mask(model_output['mask'])
#         # deaug_label = transformer.deaugment_label(model_output['label'])
#         # _ , pred = model_output.max(1)
#         # pred = torch.max(model_output,dim=1)
#         # print('pred:\n',pred)
#         # save results
#         # labels.append(deaug_label)
#         # masks.append(deaug_mask)
#
#     # reduce results as you want, e.g mean/max/min
#     # label = mean(labels)
#     # mask = mean(masks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default='senet154',help='name of deep learning net')
    parser.add_argument("--model_path",type=str,default='C:\\Users\\yxx\\Desktop\\model_senet154_ep1.pth——单模型0.541',help='path of trained net weights')
    parser.add_argument("--test_directory",type=str,default='C:\\Users\\yxx\\Desktop\\test\\TUPIAN\\\wewe',help='path of testing data')
    parser.add_argument("--dir2label_dict",type=dict,default={'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,'17':16,'18':17,'19':18,
                                                              '20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,'29':28},help='network used during the training process')
    parser.add_argument("--use_gpu", type=bool, default=False, help="weather to use gpu")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--img_resize", type=int, default=369, help="size of each image dimension")
    parser.add_argument("--img_random_crop", type=int, default=336, help="size of each image dimension")
    parser.add_argument("--predict_mode", type=str, default='pre_defined_tta', help="test time augmentation")
    parser.add_argument("--cuda_id", type=str, default='0', help="which cuda used to run the code")
    opt = parser.parse_args()
    parser.add_argument("--num_classes", type=int, default=len(opt.dir2label_dict), help="number of class")
    opt = parser.parse_args()
    net = load_model(opt.model_name,opt.model_path,opt.num_classes)
    run(net,opt)
