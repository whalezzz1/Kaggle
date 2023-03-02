import time
import datetime
from utils.utils_ import log_string
from model.model_ import *
from model.TBSCN import *
from utils.utils_ import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, args, log, loss_criterion, optimizer, scheduler):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(args)  # 加载数据
    num_train, _, num_vertex = trainX.shape  # 获取训练样本的三个维度，由于我们不需要使用第二维的大小，因此用下划线占位
    log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)  # 36458/16(batch_size)=2279t_n_b批
    val_num_batch = math.ceil(num_val / args.batch_size)  # 325批

    wait = 0
    val_loss_min = float('inf')  # 初始化为正无穷大，表示这个变量会被后面的代码用来保存验证集的最小loss值。在后续的代码中，如果验证集的loss小于当前保存的最小loss，就会更新 val_loss_min 的值，以便在之后保存最好的模型参数
    best_model_wts = None  # None 是一个特殊的数据类型，表示空值或缺少值
    train_total_loss = []  # 训练总共损失列表
    val_total_loss = []  # 验证总损失列表

    # Train & validation
    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        # shuffle
        permutation = torch.randperm(num_train)  # torch.randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        # 分别将trainX，trainTE，trainY按照相同打乱的顺序重新排列所有样本
        # train
        start_train = time.time()  # 记录开始训练的当前时间戳
        model.train()
        train_loss = 0
        for batch_idx in range(train_num_batch):  # batch_idx表示当前第几批（batch_size=16时，总共train_num_batch=2279），batch_idx从零开始
            start_idx = batch_idx * args.batch_size  # 每个batch_size的第一个样本坐标
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)  # 每个batch_size的最后一个样本坐标
            X = trainX[start_idx: end_idx]  # (16,12,325) # 切片索引前闭后开
            TE = trainTE[start_idx: end_idx]  # (16,24,2)
            label = trainY[start_idx: end_idx]  # (16,12,325)
            optimizer.zero_grad()  # 梯度归零
            ###########################################################################
            pred = model(X, TE, device).to(torch.device("cpu"))
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, label)  # 此处改进想法 不同时间片分别各自求误差分别反向传播
            train_loss += float(loss_batch) * (end_idx - start_idx)  # 此处有疑问，loss_batch不是（16，12，325）进行计算的吗为什么此处还要×16 # ************???????????????????????????????????
            loss_batch.backward()
            optimizer.step()
            if (batch_idx + 1) % 5 == 0:  # 每训练5个batch就输出一次
                print(f'Training batch: {batch_idx + 1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
            del X, TE, label, pred, loss_batch
        train_loss /= num_train  # *******************************************************************???????????????????????????????????
        train_total_loss.append(train_loss)  # *******************************************************************???????????????????????????????????
        end_train = time.time()

        # val loss
        start_val = time.time()
        val_loss = 0

        model.eval()  # 评估模式
        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx: end_idx]
                TE = valTE[start_idx: end_idx]
                label = valY[start_idx: end_idx]
                ##########################################################################
                pred = model(X, TE, device).to(torch.device("cpu"))
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += loss_batch * (end_idx - start_idx)
                del X, TE, label, pred, loss_batch
        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()
        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}')
            wait = 0
            val_loss_min = val_loss
            best_model_wts = model.state_dict()
        else:
            wait += 1
        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_file)
    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')
    return train_total_loss, val_total_loss
