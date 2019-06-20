import xlrd
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import multiprocessing
from sklearn.metrics import r2_score


#训练
def train(rank):
    floder=str(rank)+'阶'+str(time.time()).split('.')[1]
    os.chdir(os.path.join(os.getcwd(), 'output'))
    os.mkdir(floder)
    os.chdir(os.path.join(os.getcwd(),floder))

    workbook = xlrd.open_workbook(r'E:\file\AIDesign\exlinear.xlsx')
    # 根据sheet索引或者名称获取sheet内容
    sheet1 = workbook.sheet_by_index(0)  # sheet索引从0开始

    # 获取整行和整列的值（数组）
    X = sheet1.col_values(0)  # 获取第四行内容
    Y = sheet1.col_values(1)  # 获取第三列内容
    X.pop(0)
    Y.pop(0)
    X = np.array(X).reshape([len(X), 1])
    Y = np.array(Y).reshape([len(Y), 1])

    W = np.zeros((rank+1, 1))


    process = multiprocessing.Process(target=process_data, args=(X, Y, rank, W))
    process.start()


#权重迭代
def process_data(X, Y, rank, W):
    filename = str(rank)+".txt"
    file = open(filename, 'w')
    iter = 0
    H = np.ones((Y.size, 1))
    Lweight = 1
    persent = 0.1
    step_length = 1e-4*pow(0.1, rank)
    #multiple = 1000
    dw = np.zeros((rank +1, 1))
    temp_X = np.ones(( X.size, 1))
    losslist=[]

    for i in range(rank):
        i += 1
        result = pow(X, i)
        temp_X = np.hstack((temp_X, result))
    X = temp_X

    while (True):
        loss = 0
        iter += 1
        #for i in range(H.size):
        #    H[i, 0] = np.dot(W, np.transpose(X[:, i]))
        #    loss += pow((H[i, 0] - Y[i, 0]), 2)
        H = np.dot(X, W)
        loss = np.sum(pow(H - Y, 2))
        loss = loss / 2
        message='第' + str(iter) + '次迭代. 损失值：' + str(loss) + '   权值矩阵    ' + str(np.transpose(W))
        file.write(message + '\n')
        print(message)

        if iter<10:
            losslist.append(loss)
        else:
            first= losslist.pop(0)
            losslist.append(loss)
            last=loss
            absdif=abs(first-last)
            if absdif<1e-10:
                print('train finished')
                break

        for j in range(W.size):
            sum = 0
            for i in range(H.size):
                sum += (H[i, 0] - Y[i, 0]) * X[i, j]
            if W[j, 0] < 0:
                dw[j, 0] = sum + Lweight * (-persent + 2 * (1 - persent) * W[j, 0])
            else:
                dw[j, 0] = sum + Lweight * (persent + 2 * (1 - persent) * W[j, 0])

        for j in range(W.size):
            #W[0, j] = W[0, j] - step_length*(int(multiple/iter)+1)*dw[0, j]
            W[j, 0] = W[j, 0] - step_length * dw[j, 0]
    file.close()


#检测
def detect():
    weight_path=os.path.join(os.getcwd(),'output','3阶', 'weight.txt')
    data_path=os.path.join('E:/file/AIDesign', 'exlinear_test.xlsx')
    file = open(weight_path, 'r')
    message = file.read()
    weigths = message.split(' ')
    weight = []
    for w in weigths:
        weight.append(np.float64(w))
    weigth = np.array(weight).reshape([len(weight), 1])

    data = xlrd.open_workbook(data_path)
    sheet1 = data.sheet_by_index(0)
    features = sheet1.col_values(0)
    features.pop(0)
    features = np.array(features).reshape(len(features), 1)
    labels = sheet1.col_values(1)
    labels.pop(0)
    X =np.ones([len(features), 1])
    for i in range(1, len(weigth)):
        X = np.hstack((X, pow(features, i)))

    Y = np.dot(X, weigth)
    R=r2_score(labels, Y)
    plt.text( 1,2,'R^2 :'+str(R), size=13)
    #plt.plot(features, Y, 'r--')
    plt.plot(features, labels, 'g--')
    plt.show()
    print(R)
    message='------------------------\n\rR^2：'+str(R)


#主函数入口
if __name__=='__main__':

    #train(7)
    detect()
