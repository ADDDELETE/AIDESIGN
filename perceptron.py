import cv2
import os
import numpy as np
import sklearn.preprocessing as skp
import multiprocessing


#计算总图像面积与最大像素宽度
def caculate_sum_acreage_max_length(img, label):
    sum_acreage = 0
    for i in range(img.shape[0]):
        max_length = -1
        index_start = -1
        index_end = -1
        if label == 1:
            for j in range(img.shape[1]):
                if img[i, j][0] == 1 and index_start == -1:
                    index_start = j
                else:
                    index_end = j
        else:
            for j in range(img.shape[1]):
                if img[i, j][0] == 1 and index_start == -1:
                    index_start = j
                elif img[i, 27-j][0] == 1 and index_end == -1:
                    index_end = 27-j
        single_acreage = index_end-index_start
        if single_acreage != 0:
            sum_acreage += (single_acreage+1)
            if single_acreage > max_length:
                max_length = single_acreage

    return sum_acreage, max_length


#计算总像素点数
def caculate_sum_pix(img):
    sum_pix = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            if img[i, j][0] == 1:
                sum_pix += 1
    return sum_pix


#计算对角最大像素长度
def caculate_angle_max_length(img):
    index_start = 0
    index_end = 0
    for i in range(img.shape[0]):
        if img[i, i][0] == 255 and index_start == 0:
            index_start = i
        if img[27-i, 27-i][0] == 255 and index_end == 0:
            index_end = 27 - i
    max_length = index_start - index_end + 1
    return int(pow(max_length, 0.5))


#数据归一化与标准化
def normalization_and_standardization(feature):

    minMax = skp.MinMaxScaler()
    feature = minMax.fit_transform(feature)
    std = skp.StandardScaler()
    feature = std.fit_transform(feature)
    return feature


#计算单个图像图像特征值
def Process_Image(part_of_filenames):
    sum_acreage = []
    max_lengths = []
    sum_pix = []
    labels = []
    max_angle_lengths = []
    for filename in part_of_filenames:
        label = filename.split('_')[1].split('.')[0]
        img = cv2.imread(filename)
        acreage, max_length = caculate_sum_acreage_max_length(img, int(label))
        max_angle_length = caculate_angle_max_length(img)
        max_angle_lengths.append(max_angle_length)
        sum_acreage.append(acreage)
        max_lengths.append(max_length)
        sum_pix.append(caculate_sum_pix(img))
        labels.append(label)
    sum_acreage = np.array(sum_acreage).reshape([len(sum_acreage), 1])
    max_lengths = np.array(max_lengths).reshape([len(max_lengths), 1])
    max_angle_lengths = np.array(max_angle_lengths).reshape([len(max_angle_lengths), 1])
    sum_pix = np.array(sum_pix).reshape([len(sum_pix), 1])
    feature = np.hstack((sum_acreage, max_lengths, sum_pix, max_angle_lengths))
    #feature = max_lengths
    feature = normalization_and_standardization(feature)
    labels = np.array(labels).reshape([len(labels), 1])
    return feature, labels


#获得特征值矩阵
def get_feature_array(root_path,):
    K = 10
    labels = []
    feature = []
    os.chdir(root_path)
    filenames = os.listdir()
    p = multiprocessing.Pool(K)
    step_length = int(len(filenames)/K)
    result = []
    for i in range(K):
        if i != K - 1:
            result.append(p.apply_async(Process_Image, (filenames[i*step_length:(i+1)*step_length], )))
        else:
            result.append(p.apply_async(Process_Image, (filenames[i * step_length: len(filenames)],)))
    p.close()
    p.join()
    for i in range(len(result)):
        if i == 0:
            feature = result[i]._value[0]
            labels = result[i]._value[1]

        else:
            feature = np.vstack((feature, result[i]._value[0]))
            labels = np.vstack((labels, result[i]._value[1]))

    return feature, labels


#激活函数
def activate_fun(H):
    Y = []
    for i in range(H.shape[0]):
        if H[i, 0] < 0:
            Y.append(-1)
        else:
            Y.append(1)
    Y = np.array(Y).reshape(len(Y), 1)
    return Y


#获得希望输出
def get_hope_output(label):
    output = []
    for i in range(label.shape[0]):
        if label[i, 0] == '0':
            output.append(np.int32(-1))
        else:
            output.append(np.int32(1))
    output = np.array(output).reshape([len(output), 1])
    return output


#结果检验
def check_result(F, Y):
    error_list = []
    for i in range(Y.shape[0]):
        if F[i, 0] != Y[i, 0]:
            error_list.append(i)
    error_list = np.array(error_list).reshape(len(error_list), 1)
    return error_list


#计算损失值
def Loss(errorList, Y, H):
    loss = 0
    for i in range(errorList.shape[0]):
        loss += -Y[errorList[i, 0], 0] * H[errorList[i, 0], 0]
    return loss


#更新权重
def updateweight(percentage, errorlist, weight, Y, X):
    correction = np.random.randint(0, errorlist.size, int(errorlist.size / 10) + 1)
    for i in correction:
        k = errorlist[i]
        for j in range(weight.size):
            weight[j, 0] = weight[j, 0] + percentage*Y[k, 0] * X[k, j]
    return weight


#保存权重文件
def save(weight):
    file = open('E:/file/Caffe_Mnist/newfile/weight1.txt','w')
    weights=''
    for i in range(len(weight)):
        weights += (str(weight[i, 0])+' ')
    file.write(weights)
    file.close()


#加载权重文件
def load_weight(weight_path):
    file = open(weight_path, 'r')
    weights = file.read().split(' ')
    weight = []
    for i in range(len(weights)-1):
        weight.append(np.float64(weights[i]))
    weight = np.array(weight).reshape([len(weight), 1])
    file.close()
    return weight


#训练
def train():
    root_path = 'E:/file/Caffe_Mnist/newfile/image_train'
    percentage = 0.001
    iter = 0
    print('get_feature_array...')
    feature, label = get_feature_array(root_path)
    Y = get_hope_output(label)
    feature = np.hstack((feature, np.ones([feature.shape[0], 1])))
    H = np.zeros([feature.shape[0], 1])
    weight = np.ones([feature.shape[1], 1])
    print('start iteration')
    while(True):
        iter += 1
        for i in range(H.shape[0]):
            H[i, 0] = np.dot(feature[i, :], weight)
        F = activate_fun(H)
        errorList = check_result(F, Y)
        loss = Loss(errorList, Y, H)
        print('第'+str(iter)+'次迭代。 '+'损失值：'+str(loss))
        if loss == 0:
            save(weight)
            break
        weight = updateweight(percentage, errorList, weight, Y, feature)


#检测
def detect():
    root_path = 'E:/file/Caffe_Mnist/newfile/image_test'
    print('get_feature_array...')
    feature, label = get_feature_array(root_path)
    Y = get_hope_output(label)
    feature = np.hstack((feature, np.ones([feature.shape[0], 1])))
    H = np.zeros([feature.shape[0], 1])
    print('load weight')
    weight = load_weight('E:/file/Caffe_Mnist/newfile/weight1.txt')
    for i in range(H.shape[0]):
        H[i, 0] = np.dot(feature[i, :], weight)
    F = activate_fun(H)
    file = open('E:/file/Caffe_Mnist/newfile/result.txt','w')
    sum_result = ''
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(H.shape[0]):
        if F[i, 0] ==1 and Y[i, 0]==1 :
            TP += 1
            result = str(Y[i, 0]) + '  ' + str(H[i, 0])+'  '+str(F[i, 0])+' '+str(True)
        elif  F[i, 0] ==-1 and Y[i, 0]==-1:
            TN+=1
            result = str(Y[i, 0]) + '  ' + str(H[i, 0]) + '  ' + str(F[i, 0]) + ' ' + str(True)
        elif F[i, 0] ==-1 and Y[i, 0]==1:
            FN+=1
            result = str(Y[i, 0]) + '  ' + str(H[i, 0]) + '  ' + str(F[i, 0]) + ' ' + str(False)
        elif F[i, 0] ==1 and Y[i, 0]==-1:
            FP +=1
            result = str(Y[i, 0]) + '  ' + str(H[i, 0]) + '  ' + str(F[i, 0]) + ' ' + str(False)
        sum_result += (result + '\n\r')
    P = float(TP) / (TP + FP)
    R = float(TP) / (TP + FN)
    F1 = (2 * P * R) / (P + R)
    message = '-------------------------------\n\r'+'F1 :'+str(F1)
    sum_result+=message
    file.write(sum_result)
    file.close()
    print(H.shape[0])
    print(F1)


#主函数入口
if __name__ == '__main__':
    train()
    #detect()












