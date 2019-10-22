# @Time: 2019/10/22 9:10
# @Author: jwzheng
# @Function：

def count_class_num(label_list,index_list):
    label_num_dict = {}
    for index in index_list:
        if label_list[index] not in label_num_dict:
            label_num_dict[label_list[index]] = 1
        else:
            count = label_num_dict[label_list[index]]
            count += 1
            label_num_dict[label_list[index]] = count
    return label_num_dict


# 从lst中获取index为index_list的new_list
def get_index_value(value_list,index_list):
    new_lst = []
    for index in index_list:
        new_lst.append(value_list[index])
    return new_lst