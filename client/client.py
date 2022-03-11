# train whole mobilenet model
# receive global model -> local training -> send model for global aggregation
import os
import time
import socket
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim
#from torchvision import datasets, transforms
#from torch.utils.data import Dataset, DataLoader
# from openpyxl import load_workbook
import copy
#from tqdm import tqdm

import cifar10Handler
import settings_client as settings
import socket_utils
# from mobilenetv2 import Mobilenetv2_whole
from alexnet import Alexnet_whole


device = settings.device


def adjust_learning_rate(optimizer, round, epoch, learnRate, step=40):
    if ((round+1) * (epoch+1)) % step == 0:
        learnRate = learnRate * .5
        for param_group in optimizer.param_groups:
            param_group['lr'] = learnRate
    return optimizer, learnRate


if __name__ == '__main__':
    train_loss = []
    train_acc = []
    xlsname = 'client_federatedLearning_train_perepoch_info.csv' #保存文件的名称
    xlsname3 = 'FL_perround_train_info.csv'

    client_sendSize_list = []
    client_receiveSize_list = []
    client_send_time_list = []
    client_receive_time_list = []
    client_train_wait_time = []

    data_proportion = settings.data_proportion
    prep = cifar10Handler.DataHandler.preprocess()
    batch_size = settings.TRAIN_CONFIG["batch_size"]
    train_dataloader, data_size, total_batch = cifar10Handler.DataHandler.get_train_dataloader(
        batch_size=batch_size,
        transform=prep,
        data_proportion=data_proportion) #数据集信息

    print("data size: {}, total batch: {}, len(train_dataloader): {}, batch_size: {}".format(
        data_size, total_batch, len(train_dataloader), batch_size))

    test_dataloader = cifar10Handler.DataHandler.get_test_dataloader(
        transform=prep,
        batch_size=batch_size)
    validation_data = cifar10Handler.DataHandler.get_validation_data(transform=prep, n_samples=batch_size)
#验证集信息
    train_model = Alexnet_whole().to(device)#训练模型
    # print("Whole model: ", train_model)

    # Open the client socket
    s = socket.socket()
    s.connect((settings.host, settings.port))
    # server_weights, rece_w_size, rece_w_time = socket_utils.recv_gpumsg(s)
    # train_model.load_state_dict(server_weights)
    # client_receive_time_list.append(rece_w_time)
    # client_receiveSize_list.append(rece_w_size)

    criterion = nn.CrossEntropyLoss() #loss函数
    lr = settings.TRAIN_CONFIG["lr"]
    momentum = settings.TRAIN_CONFIG["momentum"]
    weight_decay = settings.TRAIN_CONFIG["weight_decay"]
    optimizer = optim.SGD(
        params=train_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)#优化器

    train_info = {
        'train_data_len': data_size,
        'client_name': settings.TRAIN_CONFIG['client_name']
    }
    send_size1, send_time1 = socket_utils.send_msg(s, train_info)
    client_sendSize_list.append(send_size1)
    client_send_time_list.append(send_time1)
    print("client is sending datasize at time {}".format(send_time1))

    startTime = time.time()
    last_wait_time = time.time()
    record_perRound_time = []
    for roundid in range(settings.TRAIN_CONFIG['rounds']):
        client_train_perRound_time = time.time()

        print("------Waiting for receiving aggregated loss-----")
        # aggregated_weights, rece_size, rece_time = socket_utils.recv_gpumsg(s)
        aggregated_weights, rece_size, rece_time = socket_utils.recv_msg(s)
        print('-----------Received aggregated weights-----------')
        client_receiveSize_list.append(rece_size)
        client_receive_time_list.append(rece_time)

        # client_train_wait_time.append(rece_time - last_wait_time)
        round_train_wait_time = rece_time - last_wait_time

        # whole_model = Mobilenetv2_whole().to(settings.device)
        d = copy.deepcopy(aggregated_weights)
        # whole_model.load_state_dict(d)
        train_model.load_state_dict(d)
        # save round model for validation
        modelSavePt = 'fl_client_alexnet_whole_model_save_round_' + str(roundid) + '_.pth'
        torch.save(train_model, modelSavePt)
        for eid in range(settings.TRAIN_CONFIG['n_epochs']):
            loss_per_batch = []
            acc_per_batch = []
            n_correct_in_epoch = 0
            startEpochTime = time.time()

            optimizer, lr = adjust_learning_rate(optimizer,roundid,eid,lr,settings.TRAIN_CONFIG["lr_step"])

            train_model.train()
            for k, data in enumerate(train_dataloader):
                x, labels = data
                x = x.to(settings.device)
                labels = labels.clone().detach().long().to(settings.device)

                optimizer.zero_grad()
                predictions = train_model(x)
                loss = criterion(predictions, labels)
                # loss.grad.item()
                loss.backward()
                optimizer.step()

                acc, n_correct = cifar10Handler.Benchmarker.get_accuracy_statistics(predictions, labels)
                loss_per_batch.append(loss.detach().item())
                acc_per_batch.append(acc)
                n_correct_in_epoch += n_correct

                print(" * batch -{:3d} || Loss={:.4f}  TrainAcc={:4.1f}%".
                      format(k, loss_per_batch[-1], acc * 100))

            e_loss = np.mean(loss_per_batch)
            e_acc = float(n_correct_in_epoch) / data_size
            train_loss.append(e_loss)
            train_acc.append(e_acc)
            endEpochTime = time.time() - startEpochTime
            print("EPOCH {:3d} || TrainAcc={:4.3f}%  T-{:.3f} LR-{:f}".format(eid, e_acc * 100, endEpochTime, lr))
            # val_loss, val_acc = cifar10Handler.Benchmarker.evaluate(
            #     train_model, *validation_data, criterion, description="validation")
            # validation_loss.append(val_loss)
            # validation_acc.append(val_acc)
            # print("EPOCH {:3d} || ValLoss={:.4f}      ValAcc={:4.1f}".format(eid, val_loss, val_acc))
            result_info = {
                'eid': [eid],
                'train loss (epoch)': [e_loss],
                'train acc (epoch)': [e_acc],
                'per epoch time': [endEpochTime]
            }
            # save result
            result_info_pd = pd.DataFrame.from_dict(result_info)
            p1 = os.path.exists(xlsname)
            if p1:  # 文件已经存在时，直接打开并追加内容，header=False
                result_info_pd.to_csv(xlsname, mode='a', index=False, header=False)
            else:  # 文件不存在时，需要添加header
                result_info_pd.to_csv(xlsname, mode='a', index=False)
            time.sleep(0.5)

        perround_elapsed_time = time.time() - client_train_perRound_time
        record_perRound_time.append(perround_elapsed_time)
        print('Finish {}-th round training with time {}'.format(roundid, perround_elapsed_time))

        print("------Update weights to server-----")
        msg = train_model.state_dict()
        send_size2, send_time2 = socket_utils.send_msg(s, msg)
        client_sendSize_list.append(send_size2)
        client_send_time_list.append(send_time2)
        last_wait_time = time.time()
        round_train_info = {
            'send model size':[send_size2],
            'receive model size': [rece_size],
            'perround time': [perround_elapsed_time],
            'wait time':[round_train_wait_time],
            'send info size':[send_size1]
        }
        round_train_pd = pd.DataFrame.from_dict(round_train_info)
        p3 = os.path.exists(xlsname3)
        if p3:  # 文件已经存在时，直接打开并追加内容，header=False
            round_train_pd.to_csv(xlsname3, mode='a', index=False, header=False)
        else:  # 文件不存在时，需要添加header
            round_train_pd.to_csv(xlsname3, mode='a', index=False)


    totalTime = time.time() - startTime
    train_model.eval()
    # torch.cuda.empty_cache()#释放显存的技巧

    bench_results = cifar10Handler.Benchmarker.run_full_benchmark(train_model)
    print('-----After {}s, Federated Learning finished with test acc {}'.
          format(totalTime, bench_results["test_acc"]))

    xlsname2 = 'train_whole_mobilenet_with_' + str(settings.TRAIN_CONFIG['n_epochs']) + '_epochs_result.csv'
    result_config = pd.DataFrame.from_dict(bench_results, orient='index')
    p2 = os.path.exists(xlsname2)
    if p2:  # 文件已经存在时，直接打开并追加内容，header=False
        result_config.to_csv(xlsname2, mode='a', index=False, header=False)
    else:  # 文件不存在时，需要添加header
        result_config.to_csv(xlsname2, mode='a', index=False)
    # if p1:  # 文件已经存在时，直接打开写入新的sheet
    #     writer_info = pd.ExcelWriter(xlsname, engine='openpyxl')
    #     book = load_workbook(writer_info.path)
    #     writer_info.book = book
    #     result_config.to_excel(excel_writer=writer_info,
    #                            sheet_name=str(settings.TRAIN_CONFIG['n_epochs']) + '_epochs_result')
    #     # device_config.to_excel(excel_writer=writer_info, sheet_name='cpu_' + cpu_percent + '_mem_' + mem_percent)
    # else:  # 文件不存在时，新建文件，并写入sheet
    #     writer_info = pd.ExcelWriter(xlsname)
    #     result_config.to_excel(excel_writer=writer_info,
    #                            sheet_name=str(settings.TRAIN_CONFIG['n_epochs']) + '_epochs_result')