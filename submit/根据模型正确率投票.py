# def get_index(lst=None, item=''):
#     return [i for i in range(len(lst)) if lst[i] == item]
#
# csv_name = [0.5288, 0.5315, 0.5304, 0.5185]
# csv1 = open('./' + str(csv_name[0]) + '.csv', 'r', encoding='utf8').readlines()
# csv2 = open('./' + str(csv_name[1]) + '.csv', 'r', encoding='utf8').readlines()
# csv3 = open('./' + str(csv_name[2]) + '.csv', 'r', encoding='utf8').readlines()
# csv4 = open('./' + str(csv_name[3]) + '.csv', 'r', encoding='utf8').readlines()
#
# csv1.pop(0)
# csv2.pop(0)
# csv3.pop(0)
# csv4.pop(0)
#
# out = []
#
# for i in range(0,len(csv1)):
#     out_line = []
#     vote1 = csv1[i].replace('\n','').split(',')
#     vote2 = csv2[i].replace('\n','').split(',')
#     vote3 = csv3[i].replace('\n','').split(',')
#     vote4 = csv4[i].replace('\n','').split(',')
#
#     out_line.append(vote1[0])
#     votes = []
#     votes.append(vote1[1])
#     votes.append(vote2[1])
#     votes.append(vote3[1])
#     votes.append(vote4[1])
#
#     beyond_one = []
#     for i in votes:
#         for j in votes:
#             if i == j:
#                 beyond_one.append(i)
#     beyond_one = list(set(beyond_one))
#
#     if len(beyond_one) == 0:
#         max_weight = max(csv_name)
#         index = csv_name.index(max_weight)
#         out.append(votes[index+1])
#     else:
#         out2 = []
#         for elem in beyond_one:
#             sorce_index = get_index(elem)
#             sorce = 0
#             for i in sorce_index:
#                 num = votes.index(i)
#                 sorce += csv_name[num + 1]
#             out2.append([elem,sorce])
#
# for


#
# file = open('./三花汇顶融合天气识别提交文件.csv', 'w', encoding='utf8')
# file.write('FileName,type\n')
# for i in range(0, len(out)):
#     # print(str(out[i]).replace('[','').replace(']','').replace('\'',''))
#     file.write(str(out[i]).replace('[','').replace(']','').replace('\'','').replace(' ','') + '\n')
# file.close()

import os
import json
# weight_dict = {'0.5288.csv':0.5288,'0.5304.csv':0.5304,'0.5315.csv':0.5315,'0.5414.csv':0.5414}
# weight_dict = {'0.5288.csv':0.5288,'0.5304.csv':0.5304,'0.5315.csv':0.5315,'0.5414.csv':0.5414,'0.5235.csv':0.5235,'0.5340.csv':0.5340}
weight_dict = {'0.5288.csv':0.5288,'0.5304.csv':0.5304,'0.5414.csv':0.5414,'0.5177.csv':0.5177,'0.5340.csv':0.5340}
def mix_res(directory):
    files = os.listdir(directory)
    total_image_label_dict = {}
    weight_list = []
    for file in files:
        weight_list.append(weight_dict[file])
        file_path = os.path.join(directory,file)
        image_label_dict = get_dict(file_path)
        for image in image_label_dict:
            if image not in total_image_label_dict:
                lst = []
                lst.append(image_label_dict[image])
                total_image_label_dict[image] = lst
            else:
                lst = total_image_label_dict[image]
                lst.append(image_label_dict[image])
                total_image_label_dict[image] = lst

    print(total_image_label_dict)
    # 开始统计
    image_label_score_dict = {}
    for image in total_image_label_dict:
        label_score_dict = {}
        label_list = total_image_label_dict[image]
        label_set = set(label_list)
        for label in label_set:
            index_list = find_index(label_list,label)
            score = get_score(weight_list,label_list,index_list)
            label_score_dict[label] = score
        image_label_score_dict[image] = label_score_dict
    print(image_label_score_dict)

    data_out = open('res.csv','w',encoding='utf8')
    for image in image_label_score_dict:
        label_score_dict = image_label_score_dict[image]
        label = get_label(label_score_dict)
        # data_out.write(image+','+json.dumps(label_score_dict)+','+str(label)+'\n')
        data_out.write(image+','+str(int(label))+'\n')
        # print(image+','+json.dumps(label_score_dict)+','+json.dumps(label_score_dict)+','+str(int(label)))
    data_out.close()


def get_label(label_score_dict):
    best_score = -99
    best_label = -99
    for label in label_score_dict:
        if label_score_dict[label] > best_score:
            best_label = label
            best_score = label_score_dict[label]
    return best_label


def get_score(weight_list,label_list,index_list):
    score = 0
    for index in index_list:
        score += weight_list[index]
    return score


def find_index(label_list,specific_label):
    index_list = []
    for i in range(0,len(label_list)):
        if specific_label==label_list[i]:
            index_list.append(i)
    return index_list


def get_dict(file_in):
    data_in = open(file_in,'r',encoding='utf8')
    data_in.readline()
    image_label_dict = {}
    for line in data_in:
        elems = line.strip().split(',')
        image_label_dict[elems[0]] = float(elems[1])
    data_in.close()
    return image_label_dict


if __name__ == '__main__':
    mix_res('res_2')