import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

alpha=1.0
round_nearest=8
data_proportion = 0.8
out_path = "./result"
root_path = '../../models/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 200

TRAIN_CONFIG = dict()
TRAIN_CONFIG["lr"] = 1e-1  # 1e-1 * 2
TRAIN_CONFIG["lr_step"] = 40
TRAIN_CONFIG["momentum"] = 0.9
TRAIN_CONFIG["weight_decay"] = 1e-5
TRAIN_CONFIG["n_epochs"] = 100

