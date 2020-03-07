import numpy as np
from random import shuffle


class Competitive_network:
    def __init__(self, input_dim, output_dim, study_speed):
        '''竞争学习神经网络初始化'''
        self.weight = np.random.rand(output_dim, input_dim)         # 权重矩阵
        self.study_speed = study_speed                              # 学习速率
        self.output = []                                            # 输出矩阵
        self.type = []                                              # 类别矩阵
        # 把权重矩阵的所有值设为0.5
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                self.weight[i][j] = 0.5

    def forward_propagation(self, x):
        '''激活和相似匹配
        返回值argmin:被激活的输出神经元的序号
        '''
        # 清除前一次的输出结果
        self.output = []
        # 计算每个输出神经元的欧几里得距离并把结果存放进输出矩阵中
        for i in range(len(self.weight)):
            mid_output = 0
            for j in range(len(self.weight[0])):
                mid_output = mid_output + pow(x[j] - self.weight[i][j], 2)
            self.output.append(np.sqrt(mid_output))
        # 获取输出矩阵中欧几里得距离最小的神经元的序号（0、1、2三者其中一个）
        argmin = np.argmin(self.output)
        return argmin

    def change_weight(self, x, argmin):
        '''更新突触权重
        x为一个训练样本
        argmin：欧氏距离最小的输出神经元序号
        '''
        # 更新欧几里得距离最小的输出神经元里的每个权重
        for i in range(len(x)-1):
            delta_weight = self.study_speed * (x[i] - self.weight[argmin][i])
            self.weight[argmin][i] = self.weight[argmin][i] + delta_weight

    def training(self, training_data):
        '''训练竞争学习的神经网络模型
        training_data: 训练数据集
        '''
        training_num = 1000      # 迭代次数
        # 计算每个输出神经元的欧几里得距离并返回欧几里得距离最小的输出神经元序号
        for i in range(training_num):
            argmin = self.forward_propagation(training_data[i % len(training_data)])
            self.change_weight(training_data[i % len(training_data)],argmin)
        # 用于统计分出的3个类中每个类对于Iris-Setosa,Iris-Versicolour和Iris-Virginica出现次数
        type_number = np.zeros((3, 3))
        # 重新计算每个输出神经元的欧几里得距离并返回欧几里得距离最小的输出神经元序号，同时更改对应类中对应花类别的出现次数
        for i in range(training_num):
            number = self.forward_propagation(training_data[i % len(training_data)])
            type_number[number][training_data[i % len(training_data)][len(training_data[0])-1]] = type_number[number][training_data[i % len(training_data)][len(training_data[0])-1]] + 1
        # 统计每个神经元出现最多次的类别
        for i in range(len(type_number)):
            self.type.append(np.argmax(type_number[i]))

    def checking(self, checking_data):
        '''用测试集来检验该模型的成功率
        checking_data: 测试数据集
        '''
        # 获取测试集长度
        checking_num = len(checking_data)
        # 正确次数初始化为0
        success_rate = 0
        # 用测试集放入竞争学习神经网络模型中去，获得分出的类别，用该分出的类别与测试集实际类对比，若一样则正确次数加1
        for i in range(checking_num):
            result = self.forward_propagation(checking_data[i])
            if self.type[result] == checking_data[i][len(checking_data[i])-1]:
                success_rate = success_rate + 1
        # 计算竞争学习神经网络的成功率并输出结果
        success_rate = success_rate * 1.0 / checking_num
        print('正确率:%.2f'%(success_rate*100),'%')

# 数据归一化
def init_input_data(all_data):
    '''
    all_data：未归一化的数据集
    返回final_input：归一化后的数据集
    '''
    max_input = np.zeros((1, len(all_data[0])-1))   # 每个属性中的最大值
    min_input = np.zeros((1, len(all_data[0])-1))   # 每个属性中的最小值
    # 获取all_data中前四个属性的最大值和最小值
    for i in range(len(all_data[0])-1):
        for j in range(len(all_data)):
            if j == 0:
                max_input[0][i] = all_data[j][i]
                min_input[0][i] = all_data[j][i]
            else:
                if max_input[0][i] < all_data[j][i]:
                    max_input[0][i] = all_data[j][i]
                if min_input[0][i] > all_data[j][i]:
                    min_input[0][i] = all_data[j][i]
    real_output = []  # 存放为归一化鸢尾类别的所有值
    # 获取归一化鸢尾类别的所有值
    for k in range(len(all_data)):
        if k == 0:
            real_output.append(all_data[k][len(all_data[0]) - 1])
        else:
            isStr = 0
            for l in range(len(real_output)):
                if real_output[l] != all_data[k][len(all_data[0]) - 1]:
                    isStr = isStr + 1
            if isStr == len(real_output):
                real_output.append(all_data[k][len(all_data[0]) - 1])
    final_input = []   # 归一化后的数据集
    # 用(x-min)/(max-min)计算鸢尾前四个属性数据的归一化，同时用类别出现的顺序归一化鸢尾类别
    for a in range(len(all_data)):
        mid_input = []
        for b in range(len(all_data[0])):
            if b < len(all_data[0]) - 1:
                mid_input.append((all_data[a][b] - min_input[0][b]) / (max_input[0][b] - min_input[0][b]))
            else:
                for c in range(len(real_output)):
                    if real_output[c] == all_data[a][b]:
                        mid_input.append(c)
        final_input.append(mid_input)
    return final_input


if __name__ == '__main__':
    data = []
    for line in open("data.txt", "r"):    # 设置文件对象并读取每一行文件
        data.append(line)                 # 将每一行文件加入到list中
    str = ''                              # 用于list化的空字符串
    data_line = []                        # 存每一行数据
    all_data = []                         # 未归一化的所有数据
    one_data = []                         # 归一化后的所有数据
    data_str = list(str)                  # 将空的str字符串list化
    # 把list形式的data转换为数组形式的all_data
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
    training_data = []                    # 训练数据
    checking_data = []                    # 测试数据
    shuffle(all_data)                     # 随机打乱数据集
    one_data = init_input_data(all_data)  # 把数据集归一化
    # 把数据集中前100个数据作为训练集，后50个数据作为测试集
    for i in range(int(len(one_data)*2/3)):
        training_data.append(one_data[i])
    for i in range(int(len(one_data)*1/3)):
        checking_data.append(one_data[int(len(one_data)*2/3)+i])
    input_dim = 4                         # 输入神经元数
    output_dim = 3                        # 输出神经元数
    study_speed = 0.1                     # 学习速率
    # 初始化竞争学习的神经网络
    network = Competitive_network(input_dim, output_dim, study_speed)
    # 用训练集训练竞争学习的神经网络
    network.training(training_data)
    # 用测试集检验竞争学习的神经网络
    network.checking(checking_data)
