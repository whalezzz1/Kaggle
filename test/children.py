from model.model_ import GMAN
import argparse
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters, load_data
from model.model_ import GMAN
from model.train import train
from model.test import test

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5, help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12, help='history steps')
parser.add_argument('--num_pred', type=int, default=12, help='prediction steps')
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7, help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1, help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--max_epoch', type=int, default=5, help='epoch to run')
parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')
parser.add_argument('--traffic_file', default='../data/pems-bay.h5', help='traffic file')
parser.add_argument('--SE_file', default='../data/SE(PeMS).txt', help='spatial embedding file')
parser.add_argument('--model_file', default='../data/GMAN.pkl', help='save the model to disk')
parser.add_argument('--log_file', default='../data/log', help='log file')
args = parser.parse_args()

# build a log file
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])

# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(args)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')
del trainX, trainTE, valX, valTE, testX, testTE, mean, std

log_string(log, 'compiling model...')
model = GMAN(SE, args, bn_decay=0.1)  # 模型初始化（）
model = model.cuda()  # 模型送到GPU里
loss_criterion = nn.MSELoss()  # 定义损失函数
loss_criterion = loss_criterion.cuda()  # 将损失函数送入GPU
optimizer = optim.Adam(model.parameters(), args.learning_rate)  # 学习速率自适应的深度神经网络方法，Adam优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.9)  # 调整学习率的机制
parameters = count_parameters(model)  # 计算可训练参数量
log_string(log, 'trainable parameters: {:,}'.format(parameters))

for sub_moudle in model.children():
    print(sub_moudle)
