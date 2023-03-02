import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / (label + 1e-8)  # 加上一个小的常数，以避免除数为零的情况
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask

    # 检查mape是否有非法值
    mape_mask = torch.isnan(mape)
    if torch.sum(mape_mask) > 0:
        # 记录非法值的位置
        mape_idx = torch.nonzero(mape_mask, as_tuple=True)
        # 检查pred和label在非法值位置的取值情况
        print(f'Illegal values found at: {mape_idx}')
        print(f'pred values: {pred[mape_idx]}')
        print(f'label values: {label[mape_idx]}')
        # 将非法值赋为0
        mape[mape_mask] = 0

    mape_mean = torch.mean(mape)

    return mae, rmse, mape_mean


"""def metric(pred, label):
    mask = torch.ne(label, 0)  # 掩码 mask 的元素值为 1 表示对应位置上标签数据不为 0，为 0 则表示对应位置上的标签数据为 0
    mask = mask.type(torch.float32)  # 将 mask 中的布尔值转化为浮点数类型（即 True 变为 1.0，False 变为 0.0）
    mask /= torch.mean(mask)  # 用整个mask的平均值进行归一化处理，这样在计算指标时就能忽略填充值对指标的影响/并将mask除以mask中非零值的平均值，以保证mask总和为1
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)  # 先求差，再求绝对值
    rmse = mae ** 2  # 差的平方（也即绝对值的平方）
    mape = mae / label  # 预测值和真实值之差的绝对值与真实值之比的平均数
    mae = torch.mean(mae)  # 一个值，平均对对误差
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))  # 这段代码首先将掩码与RMSE相乘，这样掩码为0的位置的数据点在计算RMSE时会被排除。然后，计算所有数据点的平均RMSE，并对其进行平方根运算，得到最终的RMSE值。
    mape = mape * mask
    mape = torch.mean(mape)
    return mae, rmse, mape"""


def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y


def load_data(args):
    # Traffic
    df = pd.read_hdf(args.traffic_file)
    traffic = torch.from_numpy(df.values)
    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = traffic[: train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding
    with open(args.SE_file, mode='r') as f:  # SE.txt文件中是325个顶点的向量表示，每行为一个顶点的向量
        lines = f.readlines()
        temp = lines[0].split(' ')  #
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)  # 初始化空间嵌入矩阵（全零）325×64
        for line in lines[1:]:
            temp = line.split(' ')  # 第一行将字符串切片 返回一个列表
            index = int(temp[0])  # temp[0]为节点id，后面为长度64的向量元素
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])  # 将空间节点向量填入SE矩阵325×64

    # temporal embedding
    time = pd.DatetimeIndex(df.index)  # df为数据文件pems-bay.h5，以Datetimeindex为index的Series，就是时间序列。
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                // 300
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)  # time(52116,2),52116个片的周几信息（0~6）和一天中的时间信息（0~288）
    # train/val/test
    train = time[: train_steps]  # (36481,2)
    val = time[train_steps: train_steps + val_steps]  # (5212,2)
    test = time[-test_steps:]  # (10423,2)
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, args.num_his, args.num_pred)  # tuple(2)(0(x),1(y)),分别为(36458,12,2)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)  # 将元组两个tensor在第二维度拼接为(36458,24,2)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)  # valTE(5189,24,2)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)  # testTE(10400,24,2)

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)


# dataset creation
class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# The following function can be replaced by 'loss = torch.nn.L1Loss()  loss_out = loss(pred, target)
def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)
    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)
    loss = torch.mean(loss)
    return loss


# plot train_val_loss
def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)


# plot test results
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))
