import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#host = '10.206.20.14'
host = '10.21.5.248'
port = 10086

TRAIN_CONFIG = dict()
''''''
TRAIN_CONFIG["model_name"] = "server_Alexnet"
TRAIN_CONFIG["lr"] = 1e-1  # 1e-1 * 2
TRAIN_CONFIG["lr_step"] = 40
TRAIN_CONFIG["momentum"] = 0.9
TRAIN_CONFIG["weight_decay"] = 1e-5
TRAIN_CONFIG["rounds"] = 70
TRAIN_CONFIG["users"] = 20  # number of clients

out_excel = '../mobilenet_cifar10_out/excel'
out_image = '../mobilenet_cifar10_out/image'
out_model = '../mobilenet_cifar10_out/models'

server_xlsname = 'server_client_mobilenetv2_federatedSynergy_info.csv'