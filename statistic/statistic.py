# @Time: 2019/12/11 13:43
# @Author: jwzheng
# @Function：计算多分类的混淆矩阵、recall、precision、f1
import numpy as np

# 统计多分类的混淆矩阵
def statistic(file_in):
    data_in = open(file_in,'r',encoding='utf8')
    dt = {}
    for line in data_in:
        # image_path, true_label,predict_label = line.strip().split(',')
        # true_label,predict_label = line.strip().split(',')
        filename,predict_label = line.strip().split(',')
        true_label = filename[0]
        if true_label not in dt:
            sub_dt = {}
            sub_dt[predict_label] = 1
            dt[true_label] = sub_dt
        else:
            sub_dt = dt[true_label]
            if predict_label not in sub_dt:
                sub_dt[predict_label] = 1
                dt[true_label] = sub_dt
            else:
                count = sub_dt[predict_label]
                count += 1
                sub_dt[predict_label] = count
                dt[true_label] = sub_dt
    data_in.close()
    return dt


# 统计多分类的查准率（准确率）和查全率（召回率）,f1_score
def calc_precision_recall_f1(dt):
    precision_list = []
    recall_list = []
    f1_list = []
    # 先定义一个矩阵 6*6的矩阵
    matrix = np.zeros(shape=(6,6))
    for true_label in dt:
        sub_dt = dt[true_label]
        for predict_label in sub_dt:
            matrix[int(true_label)][int(predict_label)] = sub_dt[predict_label]
    print(matrix)

    for i in range(0,len(matrix)):
        row_sum = sum(matrix[i,:])  # 某个类别真实的个数
        col_sum = sum(matrix[:,i])  # 某个类别预测的个数
        print(i,row_sum,col_sum)
        precision = matrix[i][i]/col_sum
        recall = matrix[i][i]/row_sum
        f1 = precision*recall*2/(precision+recall)
        print('class {0}, precision is {1}, recall is {2}'.format(i,precision,recall))
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list,recall_list,f1_list



if __name__ == '__main__':
    # dt = statistic('inception_resnet_v2_out.txt')
    dt = statistic('senet154_out.csv')
    print(dt)
    precision_list,recall_list,f1_list = calc_precision_recall_f1(dt)
    print('准确率是：',precision_list)
    print('召回率是：',recall_list)
    print('f1是：',f1_list)