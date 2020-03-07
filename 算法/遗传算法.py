import numpy as np
from random import shuffle
import math


# 求解二元函数3-sin^2(j*x1)-sin^2(j*x2)的min(其中j=2),即求sin^2(j*x1)+sin^2(j*x2)的max
class Genetic_algorithm:
    # 初始化种群
    def __init__(self, population_size, chromosome_length, max_value, pc, pm):
        self.population_size = population_size       # population_size为种群个体数
        self.chromosome_length = chromosome_length   # chromosome_length为染色体长度
        self.max_value = max_value                   # max_value为目标函数值域大小
        self.pc = pc                                 # pc为杂交概率
        self.pm = pm                                 # pm为变异概率
        self.population=[]                           # 种群的个体集

    # 种群的产生
    def species_origin(self):
        for i in range(self.population_size):
            temporary = []   # 染色体暂存器
            for j in range(self.chromosome_length):
                # 随机产生一个染色体,由二进制数组成
                temporary.append(np.random.randint(0, 2))
                # 把产生的染色体添加到个体集中去
            self.population.append(temporary)

    # 译码,把染色体的二进制数转换为十进制数
    def translation(self):
        temporary = []  # 存放一组染色体的x1、x2
        for i in range(len(self.population)):
            total = []
            total1 = 0
            total2 = 0
            for j in range(self.chromosome_length):
                if j < self.chromosome_length/2:
                    # 从第一个基因开始，每位对2求幂，再求和
                    total1 = total1 + self.population[i][j] * (math.pow(2, j))
                else:
                    # 从第二个基因开始，每位对2求幂，再求和
                    total2 = total2 + self.population[i][j] * (math.pow(2, j-int(chromosome_length/2)))
            total.append(total1)
            total.append(total2)
            temporary.append(total)  # 把一组x1、x2添加到temporary中
        return temporary  # 一组染色体编码完成，返回一个self.population * 2 的十进制数数组

    # 获取染色体的适应值。这里目标函数相当于环境对染色体进行筛选，这里是sin^2(j*x1)+sin^2(j*x2)
    def function(self):
        fitness_value = []  # 存放适应值
        temporary = self.translation()
        for i in range(len(temporary)):
            # x1和x2转换为从0到max_value范围里的值
            x1 = temporary[i][0] * self.max_value / (math.pow(2, int(self.chromosome_length/2)) - 1)
            x2 = temporary[i][1] * self.max_value / (math.pow(2, int(self.chromosome_length/2)) - 1)
            fitness_value.append((math.pow(math.sin(2 * x1), 2) + math.pow(math.sin(2 * x2), 2)))
        return fitness_value  # 返回适应值数组

    # 计算染色体的适应值之和
    def sum(self, fitness_value):
        total = 0
        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total

    # 计算种群中每个染色体的相对适应值
    def calculate_relative_value(self, finess_value):
        relative_value = []
        sum_value = sum(finess_value)  # 计算所有适应值之和
        for i in range(len(finess_value)):
            # 计算每个染色体的相对适应值
            value = finess_value[i] / sum_value
            relative_value.append(value)
        return relative_value

    # 计算种群中每个染色体的繁殖量（繁殖量的多少影响在杂交中被选用于杂交的概率）
    def calculate_reproduction(self, relative_value):
        reproduction = []
        for i in range(len(relative_value)):
            # 用染色体的相对适应值*种群个体数的四舍五入作为该染色体的繁殖量
            value = round(relative_value[i] * self.population_size)
            reproduction.append(value)
        return reproduction

    # 种群杂交
    def crossover(self, relative_value):
        crossover_number = round(self.pc * self.population_size)    # 该种群杂交数目
        reproduction = self.calculate_reproduction(relative_value)  # 每个染色体繁殖量
        crossover_group = []  # 按每个染色体的繁殖量扩增的种群数组，里面存放的是原种群的染色体序号
        for i in range(population_size):
            for j in range(reproduction[i]+1):
                crossover_group.append(i)
        shuffle(crossover_group)  # 把扩增的种群随机打乱顺序
        crossover = []            # 杂交染色体数组，存放真正用于繁殖的染色体序号
        number = 0                # 当前crossover_group的序号
        # 从crossover_group中选出前crossover_number个不重复的染色体序号
        while len(crossover) < crossover_number:
            if number == 0:  # 一开始把crossover_group的第一个染色体序号加入crossover中
                crossover.append(crossover_group[number])
                number = number + 1
            else:
                isNumber = 0  # 用于判断当前在crossover_group选取的染色体序号是否在crossover中存在
                for i in range(len(crossover)):
                    if crossover[i] == crossover_group[number]:
                        isNumber = 1   # 当前在crossover_group选取的染色体序号在crossover中存在则为1
                if isNumber == 0:      # 当前在crossover_group选取的染色体序号不在crossover中则加入crossover中
                    crossover.append(crossover_group[number])
                number = number + 1
        # 若当前种群杂交数为奇数个，则把crossover[0]加入crossover中防止无法两两杂交
        if (crossover_number % 2) == 1:
            crossover.append(crossover[0])
        # 从crossover中按顺序两两杂交
        for i in range(int(len(crossover)/2)):
            # 随机产生杂交位点position
            position = np.random.randint(1,self.chromosome_length-1)
            temporary1 = []   # 新产生的染色体1
            temporary2 = []   # 新产生的染色体2
            # 在两个染色体position位置断开并交换，然后放入对应的新产生的染色体中去
            temporary1.extend(self.population[crossover[i*2]][0:position])
            temporary1.extend(self.population[crossover[i*2+1]][position:self.chromosome_length])
            temporary2.extend(self.population[crossover[i*2+1]][0:position])
            temporary2.extend(self.population[crossover[i*2]][position:self.chromosome_length])
            # 把新产生的染色体放入种群个体集中去
            self.population.append(temporary1)
            self.population.append(temporary2)

    # 种群变异
    def mutation(self):
        # 获得染色体变异数目mutation_number
        mutation_number = round(self.pm * self.population_size * self.chromosome_length)
        for i in range(mutation_number):
            temporary = []  # 变异后产生的染色体
            # 随机获取变异的染色体序号population_number
            population_number = np.random.randint(0,self.population_size)
            # 随机获取需变异染色体的变异位点char_number
            char_number = np.random.randint(0,self.chromosome_length)
            for j in range(len(self.population[population_number])):
                temporary.append(self.population[population_number][j])
            # 在变异染色体的char_number位中产生变异
            temporary[char_number] = (temporary[char_number] + 1) % 2
            # 把变异染色体添加进种群个体集中
            self.population.append(temporary)

    # 种群个体筛选，选择种群中个体适应度最大的population_size个个体
    def selection(self):
        population = []  # 暂存种群个体集（每个染色体包含对应的适应值和染色体序列）
        temporary = self.population
        temporary_value = self.function()
        for i in range(len(temporary)):
            temporary0 = []
            temporary0.append(temporary_value[i])
            temporary0.append(temporary[i])
            population.append(temporary0)
        population.sort(reverse=True)  # 把个体集按适应度从大到小排序
        self.population = []           # 把原个体集初始化
        # 把暂存的个体集的前population_size个放入原个体集中
        for i in range(self.population_size):
            self.population.append(population[i][1])

    # 遗传算法的开始
    def start(self):
        iteration_number = 500  # 种群繁衍代数
        self.species_origin()   # 产生种群
        for i in range(iteration_number):
            fitness_value = self.function()  # 获取每个染色体适应值
            relative_value = self.calculate_relative_value(fitness_value)  # 获取每个染色体相对适应值
            self.crossover(relative_value)   # 种群杂交
            self.mutation()                  # 种群变异
            self.selection()                 # 种群个体筛选
        min_number = 3 - self.function()[0]  # 获取3-sin^2(j*x1)-sin^2(j*x2)的最小值
        print('3-sin^2(j*x1)-sin^2(j*x2)的最小值:%.6f'%min_number)
        min_x = self.translation()[0]        # 3-sin^2(j*x1)-sin^2(j*x2)的最小值的其中一个最优解x1和x2
        min_x1 = min_x[0] * self.max_value / (math.pow(2, int(self.chromosome_length/2)) - 1)
        min_x2 = min_x[1] * self.max_value / (math.pow(2, int(self.chromosome_length/2)) - 1)
        print('其中一个最优解x1:%.6f'%min_x1,',x2:%.6f'%min_x2)


if __name__ == '__main__':
    population_size = 50    # 种群染色体数
    chromosome_length = 46  # 染色体长度
    max_value = 6           # 目标函数值域大小
    pc = 0.8                # 杂交率
    pm = 0.01               # 变异率
    population = Genetic_algorithm(population_size, chromosome_length, max_value, pc, pm)
    population.start()