import numpy as np
import copy
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import settings_client as settings


# data_proportion = 0.2

def customizedIdxsGenerator(classes_mapper, Bias_array, local_datasize):
    """
    @Params:
    Bias_array即用于表示该客户端数据集类别偏好的数组（可能要改为字典存储）
    local_datasize用于表示该客户端有多少样本量
    classes_mapper可以放在参数表中，也可以不放(作为文件中的全局变量)
    """
    # list为可变对象，需要深拷贝
    Bias_num_array = copy.deepcopy(Bias_array)
    for i in range(len(Bias_array)):
        Bias_num_array[i] = int(local_datasize * Bias_array[i])

    local_dataitem_idxs = np.array([])
    # classes_mapper=classesIdxsGenerator(classes=classes,dataset=mnist_data_train)

    for key in classes_mapper.keys():
        # 先把classes_mapper[key] list类型转化numpy
        # classes_mapper[key]=np.array(classes_mapper[key])
        # concatenate: 拼接local_dataitem_idxs和np.random.choice
        # np.random.choice: 从classes_mapper[key]中随机抽取数字，组成int(Bias_num_array[key])大小的数组，不可以取相同数字
        local_dataitem_idxs = np.concatenate((local_dataitem_idxs,
                                              np.random.choice(classes_mapper[key], int(Bias_num_array[key]),
                                                               replace=False)), axis=0)

    np.random.shuffle(local_dataitem_idxs)
    local_dataitem_idxs = local_dataitem_idxs.astype(int)
    return local_dataitem_idxs.tolist()


# 独立同分布比较简单，直接随机抽样即可实现每个设备较为均匀的标签分布
def iid_partition(dataset, local_datasize):
    # all_clients_DatasetIdxs保存每个客户端本地数据集在整个数据集中的索引，本地训练时根据避免数据整体传输
    # all_clients_DatasetIdxs = {}
    labels_num = 10
    labels_array = [i for i in range(labels_num)]

    # 构建classes_mapper的K-V字典
    # 字典的键为标签类别，为int类型，值为该标签在整个数据集中的全部索引，为numpy类型
    classes_mapper = {}
    for item in labels_array:
        # numpy64转int
        classes_mapper[int(item)] = []
    # 根据key映射value
    for i in range(len(dataset)):
        # tensor转int
        targetForOnce = int(dataset.targets[i])
        classes_mapper[targetForOnce].append(i)
    for key in classes_mapper.keys():
        # list转为numpy数组，所以mapper的k-v类型是int-numpy_array
        classes_mapper[key] = np.array(classes_mapper[key])

    # 遍历每个客户端，为它选择它的标签分布，再获取对应的数据索引
    # for i in range(clients_num):
    # client_distribution = np.array([0.1 for i in range(labels_num)])
    client_distribution = []
    '''
    client_distribution=[0.03571428571428571, 0.14285714285714285, 0.16071428571428573, 0.07142857142857142,
                         0.16071428571428573, 0.017857142857142856, 0.14285714285714285, 0.10714285714285714,
                         0.05357142857142857, 0.10714285714285714]
    client_distribution = []
    nums = np.random.randint(1,high=11,size=labels_num) #随机数，构建labels_num维的数组
    nums_sum = np.sum(nums)
    for i in range(labels_num):
        client_distribution.append(nums[i] / nums_sum)
    '''
    iswork = True
    tmp_array = []
    each_upper = 1 / (10 * settings.data_proportion)
    print("Each upper bound is :", each_upper)
    # 均匀分布的话，每个类为0.1
    # 1/(10/2)，每两个类为一组，共5组，5组的分布和为1，每一组为0.2
    group_upper = 0.2
    check_sum = 0
    upper = each_upper if group_upper > each_upper else group_upper
    # 5个数大于0.1,5个数小于0.1
    temp_data = np.random.uniform(0.1, high=upper, size=5)
    tmp_array = np.concatenate((tmp_array, temp_data))
    for i in range(5):
        temp_data[i] = 0.2 - temp_data[i]
    tmp_array = np.concatenate((tmp_array, temp_data))
    client_distribution = np.random.choice(tmp_array, labels_num, replace=False)
    print('client distribution: ', client_distribution)
    for i in range(labels_num):
        check_sum += client_distribution[i]
        if client_distribution[i] >= (1 / (10 * settings.data_proportion)):
            iswork = False
    print("Sum of distribution is :", check_sum)
    print("Current distribution is workable (True for Yes):", iswork)

    # 根据客户端的标签分布和所需数量，根据classes_mapper获取本地数据在整个数据集中的索引
    # local_datasize = ratio * total_datesize
    client_DatasetIdxs = customizedIdxsGenerator(classes_mapper, client_distribution, local_datasize)

    # 保存到所有客户端整体的数据标签分布，以及数据索引
    # all_clients_DatasetIdxs[i] = client_DatasetIdxs

    return client_DatasetIdxs


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def localTrainLoader(indexes, data, batchSize):
    local_train_loader = DataLoader(CustomDataset(data, indexes), batch_size=batchSize, shuffle=True)

    return local_train_loader


'''
#划分训练集，不划分测试集
def main():
    transform = transforms.Compose([
        transforms.Resize((224), interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    fmnist_training_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_training_loader = DataLoader(fmnist_training_set, batch_size=100, shuffle=True, num_workers=2)


    transforms_test = transforms.Compose([
        transforms.Resize((224), interpolation=2),
        transforms.RandomGrayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    fmnist_test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms_test)
    fmnist_test_loader = DataLoader(fmnist_test_set, batch_size=50, shuffle=False, num_workers=2)


    local_datasize = data_proportion * len(fmnist_training_set)
    print('define local data size:', local_datasize)
    client_train_DatasetIdxs = iid_partition(fmnist_training_set, local_datasize)
    #print('Input client id:')
    #client_id = input() #后续可以将client的ip作为id
    #new_client = client.Client(client_id)
    # 设置数据索引的同时，对本地trainloader初始化
    local_train_set = CustomDataset(fmnist_training_set, client_train_DatasetIdxs)
    local_train_loader=localTrainLoader(client_train_DatasetIdxs,fmnist_training_set,100)
    print('------------------')
    classes = np.array(list(local_train_set.dataset.class_to_idx.values()))
    classes_test = np.array(list(fmnist_test_set.class_to_idx.values()))
    num_classes = len(classes)
    idxs = local_train_set.dataset.class_to_idx
    print("Total classes: {}".format(num_classes))
    print()
    #class_to_idx是一个字典类型
    print("class to index:{} \tType: {}".format(idxs, type(idxs)))
    print()

    #for i, data in enumerate(local_train_loader):
        # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels,data[0]表示输出数据,data[1]表示输出标签
        #print("The {}ed batch is: \n{}".format(i,data))

    #for label in idxs.keys():
        #print("The sample number of lable {} is: {}".format(label, len(local_train_set.dataset.class_to_idx[label])))
    print("Classes: {} \tType: {}".format(classes, type(classes)))
    print()
    print("Assigned training set data size:", len(local_train_set))

    #计算每一类有多少个样本
    #labels_num = 10
    #labels_array = [i for i in range(labels_num)]
    #构建classes_num的k-v字典
    classes_array = np.array([])

    #for item in labels_array:
        #classes_num[int(item)] = 0
    for batch, imgs in enumerate(local_train_loader):
        #print(imgs[1])
        classes_array=np.concatenate((classes_array,imgs[1].numpy()),axis=0)
    print("class array:{}".format(classes_array))
    #计算每一类出现多少次
    labels, counts = np.unique(classes_array, return_counts=True)
    print('{class: num}:\n',dict(zip(labels, counts)))
    print('counts sum: ', counts.sum())
    #for item in labels_array:
        #print("label {} has {} samples".format(int(item),classes_num[int(item)]))
    print("Classes Test: {} \tType: {}".format(classes_test, type(classes)))
    print("Image Shape: {}".format(local_train_set.dataset.data[0].size()))
    print("test get size:{}".format(len(local_train_loader.dataset)))
    print("Load Data Done!")

if __name__ == '__main__':
    main()
'''