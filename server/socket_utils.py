'''Socket initialization, Required socket functions'''
import struct
import pickle
import time
import torch
import io


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    # 以网络字节顺序为每条消息添加4字节长度的前缀
    # pickle.dumps(obj)：以字节对象形式返回封装的对象，不需要写入文件中；将数据通过特殊的形式转换为只有python语言认识的字符串
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    send_time = time.time()
    sock.sendall(msg)
    return l_send, send_time

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg = recvall(sock, msglen)
    msg = pickle.loads(msg)
    recv_time = time.time()
    return msg, msglen, recv_time #返回数据，数据长度

# def recv_gpumsg(sock):
#     # read message length and unpack it into an integer
#     raw_msglen = recvall(sock, 4)
#     if not raw_msglen:
#         return None
#     msglen = struct.unpack('>I', raw_msglen)[0]
#     # read the message data
#     msg = recvall(sock, msglen)
#     msg = torch.load(msg, map_location=torch.device('cpu'))
#     recv_time = time.time()
#     return msg, msglen, recv_time #返回数据，数据长度

def recvall(sock, n):#socket中没有recvall函数，需要自己实现
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

#if you are using pickle load on a cpu device, from a gpu device
#you will need to override the pickle load functionality like this:
#note: if loading directly in torch, just add map_location='cpu' in load()
#
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else: return super().find_class(module, name)

#contents = pickle.load(f) becomes... contents = CPU_Unpickler(f).load()
def recv_gpumsg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg = recvall(sock, msglen)

    msgfile_pth = "GPU_weights_msg.txt"
    msgfile = open(msgfile_pth, 'wb')
    msgfile.write(msg)
    msgfile.close()
    readmsg = open(msgfile_pth, 'rb')

    msg = CPU_Unpickler(readmsg).load()
    recv_time = time.time()
    return msg, msglen, recv_time #返回数据，数据长度