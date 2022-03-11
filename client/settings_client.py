import torch
import os,re
import psutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_realip():
    filename = "ip.swbd"
    # open(filename, "w").write("")
    os.system("ipconfig > {}".format(filename))
    text = open("{}".format(filename)).read()
    # print(text)
    try:
        ipv4 = re.findall(r'以太网适配器 以太网:(.*?)默认网关', text, re.S)[0]
        ipv4 = re.findall(r'IPv4 地址 . . . . . . . . . . . . :(.*?)子网掩码', ipv4, re.S)[0].replace(" ", "")
        print(ipv4)
    except:
        ipv4 = re.findall(r'无线局域网适配器 WLAN:(.*?)默认网关', text, re.S)[0]
        ipv4 = re.findall(r'IPv4 地址 . . . . . . . . . . . . :(.*?)子网掩码', ipv4, re.S)[0].replace(" ", "")
        print(ipv4)
    os.remove(filename)
    return ipv4


def getIP():
    """获取ipv4地址"""
    dic = psutil.net_if_addrs()
    ipv4_list = []
    for adapter in dic:
        snicList = dic[adapter]
        for snic in snicList:
            # if snic.family.name in {'AF_LINK', 'AF_PACKET'}:
            #     mac = snic.address
            if snic.family.name == 'AF_INET':
                ipv4 = snic.address
                if ipv4 != '127.0.0.1':
                    ipv4_list.append(ipv4)
            # elif snic.family.name == 'AF_INET6':
            #     ipv6 = snic.address
    if len(ipv4_list)>=1:
        return ipv4_list[0]
    else:
        return 'None'

# client_addr = get_realip() # windows适用
client_addr = getIP() # rapberryPi 适用
print("client ip:", client_addr)
alpha=1.0
round_nearest=8
data_proportion = 0.3
out_path = "../../client_federated_seynergy_result"
root_path = '../../models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#host = '10.206.20.14'
host = '10.21.5.248'
port = 10086

TRAIN_CONFIG = dict()
TRAIN_CONFIG["users"] = 2  # number of clients
TRAIN_CONFIG["local_epoch"] = 2
TRAIN_CONFIG["rounds"] = 3
TRAIN_CONFIG["batch_size"] = 512
TRAIN_CONFIG["client_name"] = 'rpi'
TRAIN_CONFIG["momentum"] = 0.9
TRAIN_CONFIG["weight_decay"] = 1e-5
TRAIN_CONFIG["lr"] = 1e-1  # 1e-1 * 2
TRAIN_CONFIG["lr_step"] = 40
TRAIN_CONFIG["n_epochs"] = 2
client_order = "0"
batch_size = TRAIN_CONFIG["batch_size"]