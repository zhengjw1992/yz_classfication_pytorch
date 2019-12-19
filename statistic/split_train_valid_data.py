# @Time: 2019/12/10 15:43
# @Author: jwzheng
# @Function：
import os
import shutil
import numpy as np


def split_train_valid_data(source_directory,save_directory,train_ratio):
    for sub_dir in os.listdir(source_directory):
        sub_dir_path = os.path.join(source_directory,sub_dir)
        file_list = os.listdir(sub_dir_path)
        np.random.shuffle(file_list)
        train_length = len(file_list)*train_ratio
        train_file_list = file_list[0:int(train_length)]
        test_file_list = file_list[int(train_length):]

        # 生成文件夹
        train_dir = os.path.join(os.path.join(save_directory,'train'),sub_dir)
        test_dir = os.path.join(os.path.join(save_directory,'test'),sub_dir)
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        # 复制文件
        Set = set()
        for file in train_file_list:
            file_path = os.path.join(sub_dir_path,file)
            print(file_path)
            shutil.copy(file_path,train_dir)
        for file in test_file_list:
            file_path = os.path.join(sub_dir_path,file)
            print(file_path)
            shutil.copy(file_path,test_dir)


if __name__ == '__main__':
    source_directory = 'C:\\Users\\yxx\\Desktop\\fog_data\\train'
    save_directory = 'C:\\Users\\yxx\\Desktop\\fog'
    train_ratio = 0.9
    split_train_valid_data(source_directory,save_directory,train_ratio)