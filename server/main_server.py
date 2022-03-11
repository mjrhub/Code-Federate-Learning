# Federated Learning server: Aggregate weights
import os
import time
import numpy as np
import copy
import torch
# from tqdm import tqdm
import socket
from threading import Thread
from threading import Lock
import socket_utils
import settings_server as settings
#from mobilenetv2 import Mobilenetv2_whole
from alexnet import Alexnet_whole

lock = Lock()
def average_weights(w, datasizelist):
    for i, data in enumerate(datasizelist):
        for key in w[i].keys():
            w[i][key] = w[i][key] * float(data)

    w_avg = copy.deepcopy(w[0])
    # when client use only one kinds of device
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasizelist)))
    return w_avg
    # when client use various devices (cpu, gpu) you need to use it instead
    # for key, val in w_avg.items():
    #     common_device = val.device
    #     break
    # for key in w_avg.keys():
    #     for i in range(1, len(w)):
    #         if common_device == 'cpu':
    #             w_avg[key] += w[i][key].cpu()
    #         else:
    #             w_avg[key] += w[i][key].cuda()
    #     w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

def train(userid, num_users, client_conn, rounds):
    global server_model_weights_list
    global client_data_size_list
    global global_weights
    global weight_count

    for r in range(rounds):
        with lock:
            if weight_count == num_users:
                modelSavePt_c = 'FL_alexnet_global_model_save_round_' + str(r) + '_.pth'
                torch.save(global_weights, modelSavePt_c)
                for i, conn in enumerate(clientsoclist):
                    datasize1, send_time1 = socket_utils.send_msg(conn, global_weights)
                    print("----sent global weights to client {} ----".format(i))
                    server_total_send_time_list[userid].append(send_time1)
                    server_total_sendsize_list[userid].append(datasize1)
                    weight_count = 0

        client_weights, datasize2, rece_time2 = socket_utils.recv_msg(client_conn)

        with lock:
            server_total_receive_time_list[userid].append(rece_time2)
            server_total_receivesize_list[userid].append(datasize2)
            server_model_weights_list[userid] = client_weights
        print("Round {}: Received client {}'s weights".format(r, userid))
        with lock:
            weight_count += 1
            if weight_count == num_users:
                # average
                global_weights = average_weights(server_model_weights_list, client_data_size_list)

def receive(userid, num_users, conn, rounds):  # thread for receive clients
    global weight_count
    global server_total_receive_time_list
    global server_total_receivesize_list
    global client_data_size_list
    global clientNameList

    #     train_info = {
    #     'train_data_len': data_size,
    #     'client_name': settings.TRAIN_CONFIG['client_name']
    # }
    client_train_info, msg_length, recieve_time = socket_utils.recv_msg(clientsoclist[userid])
    client_data_size = int(client_train_info['train_data_len'])
    client_name = client_train_info['client_name']
    print("----------------- Received clients' train_info ------------")

    with lock:
        server_total_receive_time_list[userid].append(recieve_time)
        server_total_receivesize_list[userid].append(msg_length)
        client_data_size_list[userid] = client_data_size
        clientNameList[userid] = client_name
        weight_count += 1

    # def train(userid, train_dataset_size, num_users, client_conn, rounds):
    train(userid, num_users, conn, rounds)

### global variance

clientsoclist = [0]*settings.TRAIN_CONFIG['users']
clientNameList = [0] * settings.TRAIN_CONFIG['users']
client_data_size_list = [0]*settings.TRAIN_CONFIG['users']

# 记录传送的数据量
server_total_sendsize_list = [[] for i in range(settings.TRAIN_CONFIG['users'])]
server_total_receivesize_list = [[] for i in range(settings.TRAIN_CONFIG['users'])]

# 记录通信时间
server_total_send_time_list = [[] for i in range(settings.TRAIN_CONFIG['users'])]
server_total_receive_time_list = [[] for i in range(settings.TRAIN_CONFIG['users'])]

server_model_weights_list = [0]*settings.TRAIN_CONFIG['users'] # {"client_conn": server model}
weight_count = 0

# mobilemodel = Mobilenetv2_whole()
# mobilemodel.to(settings.device)
# global_weights = copy.deepcopy(mobilemodel.state_dict())
alexmodel = Alexnet_whole()
alexmodel.to(settings.device)
global_weights = copy.deepcopy(alexmodel.state_dict())

if __name__ == '__main__':

    # addrs = socket.getaddrinfo(socket.gethostname(), None)
    # host = addrs[-1][4][0]
    host = settings.host
    port = settings.port
    print('host {}: port {}'.format(host, port))
    # 开启服务端
    sock = socket.socket()
    sock.bind((host, port))
    sock.listen(100)
    print('The number of users: {}'.format(settings.TRAIN_CONFIG['users']))

    thrs = []
    start_time = time.time()
    for userid in range(settings.TRAIN_CONFIG['users']):
        conn, addr = sock.accept()
        print('Conntected with', addr[1])
        # append client socket on list
        print(userid)
        clientsoclist[userid] = conn
        print('socket list index {} records connection with client {}'.format(userid, clientsoclist[userid]))

        # receive(userid, num_users, conn, rounds)
        args = (userid, settings.TRAIN_CONFIG['users'], conn, settings.TRAIN_CONFIG["rounds"])
        thread = Thread(target=receive, args=args)
        thrs.append(thread)
        print("start thread of client {}".format(userid))
        thread.start()

    for thread in thrs:
        thread.join()

    end_time = time.time()
    print("Federated Total Training Time: {} sec".format(end_time - start_time))
