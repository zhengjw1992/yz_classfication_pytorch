# @Time: 2019/9/30 9:47
# @Author: jwzheng
# @Function： 读取图片
import os
import torchvision


# 读取数据集 生成满足Dataset格式的数据
def read_data(directory,dir2label_dict):
    filename_list = []
    label_list = []
    sub_directory_list = os.listdir(directory)   # train_data or validation_data
    # 可以考虑将sub_directory_list写死 以指定读取的label顺序
    for sub_directory in sub_directory_list:
        sub_directory_path = os.path.join(directory,sub_directory)  # 每一个子类别
        image_list = os.listdir(sub_directory_path)
        for image in image_list:
            image_path = os.path.join(sub_directory_path,image)
            filename_list.append(image_path)
            label_list.append(dir2label_dict[sub_directory])
    return filename_list,label_list


# 数据增强操作
def data_augmentation(img_resize,img_random_crop,mode):
    if mode=='train':
        augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(img_resize),
                                                             torchvision.transforms.RandomCrop(img_random_crop),
                                                             torchvision.transforms.RandomHorizontalFlip(),
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                             ])
        return augmentation
    elif mode=='predict':
        augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(img_resize),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        return augmentation
    else:
        print('data_augmentation()的mode参数设置有误。')


