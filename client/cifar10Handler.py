#class DataHandler; class BenchMark; DataSet: cifar 10
import numpy as np
import time
from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from dataPartition import iid_partition, CustomDataset, localTrainLoader
import settings_client as settings


class DataHandler:
    '''Define local cifar10 data'''
    CIFAR10_DATA_DIR = "../data"
    N_WORKERS = 2

    @classmethod
    def preprocess(cls, mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)):
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform_train
    '''
    CIFAR10_TRAIN_MEAN = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
    CIFAR10_TRAIN_STD = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
     transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    '''

    @classmethod
    def get_train_dataloader(cls, batch_size, transform, data_proportion) -> DataLoader:
        cifar10_training_set = torchvision.datasets.CIFAR10(train=True, root=cls.CIFAR10_DATA_DIR, transform=transform, download=True)
        # cifar10_training_loader = DataLoader(cifar10_training_set, batch_size=batch_size, shuffle=True, num_workers=cls.N_WORKERS)

        # get local data set
        local_datasize = data_proportion * len(cifar10_training_set)
        print('define local data size:', local_datasize)
        print('----------------------------------------------------------------------')
        client_train_DatasetIdxs = iid_partition(cifar10_training_set, local_datasize)

        # 设置数据索引的同时，对本地trainloader初始化
        local_train_set = CustomDataset(cifar10_training_set, client_train_DatasetIdxs)
        local_train_loader = localTrainLoader(client_train_DatasetIdxs, cifar10_training_set, batch_size)
        print('----------------------------------------------------------------------')
        classes = np.array(list(local_train_set.dataset.class_to_idx.values()))
        num_classes = len(classes)
        idxs = local_train_set.dataset.class_to_idx
        print("Total classes: {}".format(num_classes))
        print()
        # class_to_idx是一个字典类型
        print("class to index:{} \tType: {}".format(idxs, type(idxs)))
        print()
        '''
            for i, data in enumerate(local_train_loader):
                # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels,data[0]表示输出数据,data[1]表示输出标签
                print("The {}ed batch is: \n{}".format(i,data))
            '''
        # for label in idxs.keys():
        # print("The sample number of lable {} is: {}".format(label, len(local_train_set.dataset.class_to_idx[label])))
        print("Classes: {} \tType: {}".format(classes, type(classes)))
        print()
        print("Assigned training set data size:", len(local_train_set))

        # 计算每一类有多少个样本
        # 构建classes_num的k-v字典
        classes_array = np.array([])
        for batch, imgs in enumerate(local_train_loader):
            # print(imgs[1])
            classes_array = np.concatenate((classes_array, imgs[1].numpy()), axis=0)
        print("class array:{}".format(classes_array))
        #print("length of class array:{}".format(len(classes_array)))
        # 计算每一类出现多少次
        labels, counts = np.unique(classes_array, return_counts=True)
        print('{class: num}:\n', dict(zip(labels, counts)))
        print('counts sum: ', counts.sum())
        # for item in labels_array:
        # print("label {} has {} samples".format(int(item),classes_num[int(item)]))
        # print("Image Shape: {}".format(local_train_set.dataset.data[0].size()))
        print("Load Training Data Done!")
        return local_train_loader, len(local_train_set), len(local_train_loader)

    @classmethod
    def get_validation_data(cls, n_samples=100, transform=None) -> (torch.Tensor, torch.Tensor):
        test_data = torchvision.datasets.CIFAR10(train=False, root=cls.CIFAR10_DATA_DIR, transform=transform,
                                                      download=True)
        dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=True, num_workers=cls.N_WORKERS)
        validation_input, validation_labels = iter(dataloader).next()
        # print(Counter(validation_labels.numpy()))
        return validation_input.to(settings.device), validation_labels.to(settings.device)

    @classmethod
    def get_test_dataloader(cls, transform, batch_size) -> (torch.Tensor, torch.Tensor):
        # prep = cls.preprocess(augmentations=False, padding_to_32=padding_to_32)
        test_data = torchvision.datasets.CIFAR10(train=False, root=cls.CIFAR10_DATA_DIR, transform=transform,
                                                      download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=cls.N_WORKERS)
        return test_loader

    @classmethod
    def get_class_names(cls):
        #  ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        data = torchvision.datasets.CIFAR10(train=True, root=cls.CIFAR10_DATA_DIR, download=True)
        return data.classes


class Benchmarker:
    """Quality evaluation and performance benchmarking."""

    @staticmethod
    def get_accuracy_statistics(predictions: torch.Tensor, labels: torch.Tensor):
        """Return accuracy metric and number of correct predictions."""
        predicted_classes = predictions.argmax(dim=1)
        assert predicted_classes.nelement() == labels.nelement(), "Predictions and Labels have different lengths!!"
        num_correct = float(sum(predicted_classes == labels.to(settings.device)))
        acc = num_correct / float(labels.nelement())
        return acc, num_correct

    @classmethod
    def evaluate(cls, model: nn.Module, input_tensor, labels, criterion, description="", verbose=True):
        model.eval()
        with torch.no_grad():
            predictions = model(input_tensor)
            loss = criterion(predictions, labels)
            acc, _ = cls.get_accuracy_statistics(predictions, labels)
            n = len(labels)
        if verbose:
            print("\n{}".format("=" * 50))
            print("[{:^17s}] || Loss={:.4f}  Acc={:4.1f}%  n={}".
                  format(description.upper(), loss.detach().item(), acc * 100, n))
        return loss, acc

    @classmethod
    def evaluate_test_data(cls, model: nn.Module, criterion):
        model.eval()

        prep_transform = DataHandler.preprocess()
        dataloaders_list = list()
        dataloaders_list.append(DataHandler.get_test_dataloader(transform=prep_transform, batch_size= settings.batch_size))
        #dataloaders_list.append(DataHandler.get_test_dataloader(transform=prep_transform, batch_size= settings.TRAIN_CONFIG["batch_size"])

        n = len(dataloaders_list[0].dataset)
        acc_list, loss_list = [], []
        with torch.no_grad():
            for dlid, data_loader in enumerate(dataloaders_list):
                num_correct = 0
                losses = []
                for test_input, test_labels in data_loader:
                    test_labels = test_labels.to(settings.device)
                    predictions = model(test_input.to(settings.device))
                    loss = criterion(predictions, test_labels)
                    losses.append(loss.detach().item())
                    _, num_correct_in_batch = cls.get_accuracy_statistics(predictions, test_labels)
                    num_correct += num_correct_in_batch
                test_loss = np.mean(losses)
                test_acc = num_correct / n
                acc_list.append(test_acc)
                loss_list.append(test_loss)
                print("[{:^17s}] || Loss={:.4f}  Acc={:4.2f}%  n={}".
                      format("TEST EVAL- [{:}]".format(dlid), test_loss, test_acc * 100, n))
        return min(loss_list), max(acc_list)

    def get_test_dataset(cls):
        # padding_to_32 = True if model.__class__.__name__ == "EfficientNet" else False
        prep_default = DataHandler.preprocess()
        dataloaders_list = list()
        dataloaders_list.append(DataHandler.get_test_dataloader(transform=prep_default, batch_size= settings.batch_size))
        #dataloaders_list.append(DataHandler.get_test_dataloader(transform=prep_transform, batch_size=settings.TRAIN_CONFIG["batch_size"])

        return dataloaders_list

    @staticmethod
    def run_net_summary(model, input_shape=(3, 32, 32)):
        summary(model, input_shape)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(model)
        return total_params

    @staticmethod
    def get_inference_time(model: nn.Module):
        batch_size = 100
        n_iters = 100
        prep = DataHandler.preprocess()
        data, dataSize, totalBatch = DataHandler.get_train_dataloader(batch_size=batch_size, transform=prep, data_proportion=settings.data_proportion)
        times_per_iter = []
        for iter_id, (x, _) in enumerate(data):
            if iter_id >= n_iters:
                break
            # https://blog.csdn.net/u013548568/article/details/81368019
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            start = time.time()
            _ = model(x.to(settings.device))
            # end.record()
            _.cpu()
            end = time.time()
            # Waits for everything to finish running
            # torch.cuda.synchronize()
            # t = start.elapsed_time(end)
            t = end - start
            times_per_iter.append(t)
        t_mean = np.mean(times_per_iter)
        t_std = np.std(times_per_iter)
        print("[{:^17s}] || t_mean={:.1f}[millisec]  t_std={:.1f}[millisec]  n_iters={}".
              format("TIMING per 1k", t_mean, t_std, n_iters))
        return t_mean, t_std

    @classmethod
    def run_full_benchmark(cls, model: nn.Module, verbose=False):
        print("\nStarting full_benchmark . . .")
        loss_criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = cls.evaluate_test_data(model, loss_criterion)
        inference_time = cls.get_inference_time(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results = OrderedDict()
        results["model"] = model.__class__.__name__
        results["test_acc"] = test_acc
        results["test_cross_entropy"] = test_loss
        results["n_params"] = n_params
        results["inference_time_mean"] = inference_time[0]
        results["inference_time_std"] = inference_time[1]
        if verbose:
            print(results)
        return results

def get_test_datasets(file):
    prep_default = DataHandler.preprocess()
    test_data = torchvision.datasets.CIFAR10(train=False, root=file, transform=prep_default,
                                                      download=True)
    test_loader = DataLoader(test_data, batch_size=100, num_workers=2)
    return test_loader