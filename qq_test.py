
import time
import os
from segmentation.VOC.voc import VOCSegmentation
import argparse
parser = argparse.ArgumentParser(description='DATA-RefineNet Training With PyTorch')
parser.add_argument('--model_name', default='DATA-RefineNet', type=str,
                    help='The model name')
parser.add_argument('--data_config', default='configs/VOC_config.yaml', 
                    metavar='FILE', help='path to data cfg file', type=str,)
parser.add_argument('--device_gpu', default='0,1,2', type=str,
                    help='Cuda device, i.e. 0 or 0,1,2,3')
parser.add_argument('--checkpoint', default='checkpoints/cnn_log_eval/eval-Gumbel_model_M4-dp0.2-20220824-09/Class_21_epoch_0.pt', help='The checkpoint path')

parser.add_argument('--save', type=str, default='checkpoints',
                    help='save model checkpoints in the specified directory')
parser.add_argument('--epochs', '-e', type=int, default=65,
                    help='number of epochs for training') #default 65
parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                    help='epochs at which to decay learning rate')
parser.add_argument('--warmup', type=int, default=None)
parser.add_argument('--seed', '-s', default = 42 , type=int, help='manually set random seed for torch')




args = parser.parse_args()
    
import yaml    
data_cfg_path = open(args.data_config)
# 引入EasyDict 可以让你像访问属性一样访问dict里的变量。
from easydict import EasyDict as edict
data_cfg = yaml.full_load(data_cfg_path)
data_cfg = edict(data_cfg) 
args.data = data_cfg
train_dataset = VOCSegmentation(args.data.DATASET_PATH, args, split = 'train')
image, label = train_dataset[0]
print('Done')




import numpy as np
import pandas as pd

image.save("test.jpg")

a = label.numpy()
pd.DataFrame(a).to_csv('sample.csv')

from PIL import Image

a = a/a.max() * 255 

im = Image.fromarray(a)
if im.mode == "F":
    im = im.convert('RGB') 
im.save("mask.jpg")