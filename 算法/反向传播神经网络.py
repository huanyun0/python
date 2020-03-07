import numpy as np
import layer
from random import shuffle
import math



INIT_EPSILON = 0.1
EPSILON = 0.0001

# hidden_dim:一层隐藏层神经元数目
# hidden_num:隐藏层层数
# input_deltas:神经元的阈值



class NeuralNetwork:
    def __init__(self):
        self.weight = []                    # 权重矩阵([1*m*n+(l-1)*m*m+1*k*m]) n为输入层顶点数，m为一层隐含层的顶点数，l为隐含层数，k为输出层顶点数
        self.layers = layer.Layer()         # 神经网络的神经元组
        self.input_deltas = []              # 阈值矩阵(l*m+1*k)
        self.input_data = []                # 输入总数据：将训练数据n维拆分为n-1维作为输入数据，第n维作为期望输出

    # 初始化权重矩阵
    def init_weight(self, input_dim, hidden_dim, hidden_num, output_dim):
        str1 = np.zeros((input_dim, hidden_dim))
        str2 = np.zeros((hidden_num-1,hidden_dim, hidden_dim))
        str3 = np.zeros((hidden_dim, output_dim))
        self.weight.append(str1)
        for i in range(hidden_num-1):
            self.weight.append(str2[i])
        self.weight.append(str3)
        for a in range(len(self.weight)):
            for b in range(len(self.weight[a])):
                for c in range(len(self.weight[a][b])):
                    self.weight[a][b][c] = np.random.uniform(-0.5,0.5) * 2 * INIT_EPSILON - EPSILON


    # 初始化神经元的阈值
    def init_input_deltas(self, hidden_dim, hidden_num, output_dim):
        str1 = np.zeros((hidden_num,hidden_dim))
        str2 = np.zeros((1,output_dim))
        for i in range(hidden_num):
            self.input_deltas.append(str1[i])
        self.input_deltas.append(str2[0])
        for a in range(len(self.input_deltas)):
            for b in range(len(self.input_deltas[a])):
                self.input_deltas[a][b] = np.random.uniform(-1,1)

    # 初始化神经网络元组里神经元类型、输入、实际输出、误差梯度和期望输出(或误差梯度*权重之和)
    def init_layers(self, input_dim, hidden_dim, hidden_num, output_dim):
        str_type = []
        str_input = []
        str_output = []
        str_error_gradient = []
        str_real_output = [0, 0, 0]

        str_type1 = np.zeros((1, input_dim))
        str_type2 = np.zeros((hidden_num, hidden_dim))
        str_type3 = np.zeros((1, output_dim))

        str_input1 = np.zeros((1, input_dim))
        str_input2 = np.zeros((hidden_num, hidden_dim))

        str_output2 = np.zeros((hidden_num, hidden_dim))
        str_output3 = np.zeros((1, output_dim))

        str_error_gradient2 = np.zeros((hidden_num, hidden_dim))
        str_error_gradient3 = np.zeros((1, output_dim))

        str_type.append(str_type1[0])
        for i in range(hidden_num):
            str_type.append(str_type2[i])
        str_type.append(str_type3[0])
        for i in range(len(str_type[0])):
            str_type[0][i] = 1
        for i in range(len(str_type[len(str_type)-1])):
            str_type[len(str_type)-1][i] = 2
        self.layers.set_type(str_type)

        str_input.append(str_input1[0])
        for i in range(hidden_num):
            str_input.append(str_input2[i])
        self.layers.set_input(str_input)

        for i in range(hidden_num):
            str_output.append(str_output2[i])
        str_output.append(str_output3[0])
        self.layers.set_output(str_output)

        for i in range(hidden_num):
            str_error_gradient.append(str_error_gradient2[i])
        str_error_gradient.append(str_error_gradient3[0])
        self.layers.set_error_gradient(str_error_gradient)

        self.layers.set_real_output(str_real_output)

    # 初始化输入总数据,把数据归一化
    def init_input_data(self, training_data):
        #max_input = np.zeros((1, len(training_data[0])-1))   # 每个属性中的最大值
        #min_input = np.zeros((1, len(training_data[0])-1))   # 每个属性值的最小值
        #for i in range(len(training_data[0])-1):
            #for j in range(len(training_data)):
                #if j == 0:
                    #max_input[0][i] = training_data[j][i]
                    #min_input[0][i] = training_data[j][i]
                #else:
                    #if max_input[0][i] < training_data[j][i]:
                        #max_input[0][i] = training_data[j][i]
                    #if min_input[0][i] > training_data[j][i]:
                        #min_input[0][i] = training_data[j][i]
        max_input = [7.9, 4.4, 6.9, 2.5]
        min_input = [4.3, 2.0, 1.0, 0.1]
        real_output = []    #存放用于期望输出且未归一化的训练数据的最后一个属性的所有值
        for k in range(len(training_data)):
            if k == 0:
                real_output.append(training_data[k][len(training_data[0])-1])
            else:
                isStr = 0
                for l in range(len(real_output)):
                    if real_output[l] == training_data[k][len(training_data[0])-1]:
                        isStr = 1
                if isStr == 0:
                    real_output.append(training_data[k][len(training_data[0])-1])
        # 归一化
        for a in range(len(training_data)):
            mid_input = []
            for b in range(len(training_data[0])):
                if b < len(training_data[0])-1:
                    mid_input.append((training_data[a][b]-min_input[b])/(max_input[b]-min_input[b]))
                else:
                    for c in range(len(real_output)):
                        if real_output[c] == training_data[a][b]:
                            mid_input.append(c)
            self.input_data.append(mid_input)

    # 基于反向传播的神经网络
    def back_propagation(self, training_data, hidden_dim, hidden_num, output_dim):
        if hidden_num < 1:
            print("隐藏层数目不得小于1")
        iteration_number = 40000
        input_dim = len(training_data[0]) - 1
        training_num = len(training_data)
        self.init_weight(input_dim, hidden_dim, hidden_num, output_dim)
        self.init_input_deltas(hidden_dim, hidden_num, output_dim)
        self.layers.set_input_weight(self.weight)
        self.layers.set_input_deltas(self.input_deltas)
        self.init_layers(input_dim, hidden_dim, hidden_num, output_dim)
        self.init_input_data(training_data)
        for i in range(iteration_number):
            # 前向传播
            # 第一层隐含层神经元获取输入并计算输出值
            for j in range(len(self.layers.input[0])):
                self.layers.input[0][j] = self.input_data[i%training_num][j]
            for j in range(len(self.layers.input_weight[0][0])):
                self.layers.output[0][j] = self.layers.sigmoid(0, j)
            # 之后每个隐含层神经元获取上一层的输出值并计算当前层的输出值
            for j in range(len(self.layers.input_weight)-2):
                for k in range(len(self.layers.input_weight[1][0])):
                    self.layers.input[j+1][k] = self.layers.output[j][k]
                for k in range(len(self.layers.input_weight[1][0])):
                    self.layers.output[j+1][k] = self.layers.sigmoid(j+1, k)
            # 输出层神经元获取上一层隐含层的输出值并计算输出值
            for j in range(len(self.layers.input[len(self.layers.input)-1])):
                self.layers.input[len(self.layers.input)-1][j] = self.layers.output[len(self.layers.output) - 2][j]
            for j in range(len(self.layers.input_weight[len(self.layers.input_weight)-1][0])):
                self.layers.output[len(self.layers.output)-1][j] = self.layers.sigmoid(len(self.layers.input)-1, j)
            # 反向传播
            # 计算输出层神经元的误差梯度并校正权重
            #print('第', i+1, '次迭代前:', self.layers.input_weight)
            self.layers.real_output[self.input_data[i%training_num][len(self.input_data[i%training_num])-1]] = 1
            for j in range(len(self.layers.output[len(self.layers.output)-1])):
                self.layers.caculate_error_gradient(len(self.layers.error_gradient)-1, j)
            # 计算隐含层神经元的误差梯度并校正权重
            for j in range(len(self.layers.output)-1):
                for k in range(len(self.layers.output[len(self.layers.output)-2-j])):
                    self.layers.caculate_error_gradient(len(self.layers.output)-2-j, k)
            self.layers.real_output = [0, 0, 0]
            #print('第', i+1, '次迭代后:', self.layers.input_weight)

    # 用测试数据来验证成功率
    def check(self, real_data):
        real_data_number = len(real_data)
        success_rate = 0
        self.init_input_data(real_data)
        for i in range(real_data_number):
            for j in range(len(self.layers.input[0])):
                self.layers.input[0][j] = self.input_data[i][j]
            for j in range(len(self.layers.input_weight[0][0])):
                self.layers.output[0][j] = self.layers.sigmoid(0, j)
            # 之后每个隐含层神经元获取上一层的输出值并计算当前层的输出值
            for j in range(len(self.layers.input_weight)-2):
                for k in range(len(self.layers.input_weight[1][0])):
                    self.layers.input[j+1][k] = self.layers.output[j][k]
                for k in range(len(self.layers.input_weight[1][0])):
                    self.layers.output[j+1][k] = self.layers.sigmoid(j+1, k)
            # 输出层神经元获取上一层隐含层的输出值并计算输出值
            for j in range(len(self.layers.input[len(self.layers.input)-1])):
                self.layers.input[len(self.layers.input)-1][j] = self.layers.output[len(self.layers.output) - 2][j]
            for j in range(len(self.layers.input_weight[len(self.layers.input_weight)-1][0])):
                self.layers.output[len(self.layers.output)-1][j] = self.layers.sigmoid(len(self.layers.input)-1, j)
            self.layers.real_output[self.input_data[i][len(self.input_data[i]) - 1]] = 1
            max_number = 0
            max_possibility = 0
            for j in range(len(self.layers.output[len(self.layers.output)-1])):
                if max_number < self.layers.output[len(self.layers.output)-1][j]:
                    max_number = self.layers.output[len(self.layers.output)-1][j]
                    max_possibility = j
            if self.layers.real_output[max_possibility] == 1:
                success_rate = success_rate + 1
            print('第',i+1,'次',self.layers.output[len(self.layers.output)-1],' ',self.layers.real_output)
            self.layers.real_output = [0, 0, 0]
        print('正确率:',success_rate*1.0/real_data_number)

if __name__ == '__main__':
    output_dim = 3
    data = []
    for line in open("data.txt", "r"):  # 设置文件对象并读取每一行文件
        data.append(line)  # 将每一行文件加入到list中
    str = ''               # 用于list化的空字符串
    data_line = []         # 存每一行数据
    all_data = []          # 所有数据
    data_str = list(str)   # 将空的str字符串list化
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            if data[i][j] != ',':
                data_str.insert(len(data_str), data[i][j])
            else:
                data_line.append(float(''.join(data_str)))
                data_str.clear()
        data_line.append(''.join(data_str))
        data_str.clear()
        all_data.append(data_line)
        data_line = []
    shuffle(all_data)
    training_data = []   # 训练数据
    real_data = []       # 测试数据
    for i in range(100):
        training_data.append(all_data[i])
    for i in range(50):
        real_data.append(all_data[100+i])
    network = NeuralNetwork()
    hidden_num = int(input('请输入隐含层层数:\n'))
    hidden_dim = int(input('请输入每层隐含层的神经元数:\n'))
    network.back_propagation(training_data, hidden_dim, hidden_num, output_dim)
    network.check(real_data)
