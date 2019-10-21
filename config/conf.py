# @Time: 2019/10/8 10:41
# @Author: jwzheng
# @Function：模型训练的配置参数类


# # 参数类
# class conf(object):
#     # 文件夹和真实标签的映射表
#     # dir2label_dict = {'yes':1,'no':0}
#     dir2label_dict = {'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,'17':16,'18':17,'19':18,
#                       '20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,'29':28}
#
#     # 使用的模型的名称，目前支持三个模型：resnext50，resnext101，se_resnext101
#     model_name = 'resnext50'
#     batch_size = 32  # 对各种模型都是用32来进行测试
#     epoch = 40
#     learning_rate = 1e-4
#     # 分类的类别的个数
#     num_classes = len(dir2label_dict)
#     # 训练集和验证集的文件夹路径
#     train_directory = 'mnist_pytorch/test/train_data'
#     val_directory = 'mnist_pytorch/test/validation_data'
#     # 设置模型保存的文件夹路径
#     model_save_path = 'output/'+model_name+'/model_save_dir'
#     # 设置日志保存的文件夹路径
#     log_save_path = 'output/'+model_name+'/log_save_dir'
#     # 设置使用gpu还是cpu
#     use_gpu = True
#     # 设置使用第几个显卡  支持 0,1,2,3及及其组合
#     cuda_device_num = '0'
