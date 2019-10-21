# @Time: 2019/9/30 10:42
# @Author: wanglj
# @Function： 自定义dataset
from torch.utils.data import TensorDataset,DataLoader,Dataset
from PIL import Image
# class CustomDataset(Dataset): #需要继承data.Dataset
#     # augmentation:数据增强的方式
#     def __init__(self,augmentation):
#         self.aug = augmentation
#         # TODO
#         # 1. Initialize file path or list of file names.
#         pass
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         #这里需要注意的是，第一步：read one data，是一个data
#         pass
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return 0


#filenames是训练数据文件名称列表，labels是标签列表
class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx] - 1