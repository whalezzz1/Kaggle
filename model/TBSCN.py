import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_comput import attention_f
import math


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,  # 核心
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)  #
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)  # **作用：**卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    """
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    """

    def __init__(self, D, bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, T=288):  # TE（16，24，2）
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)  # (325,64)--->(1,1,325,64)
        SE = self.FC_se(SE)  # (1,1,325,64)--->(1,1,325,64)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)  # (16,24,7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)  # (16,24,288)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)  # (16,24,7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)  # # (16,24,288)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # TE(16,24,295)one hot时间嵌入
        TE = TE.unsqueeze(dim=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TE = TE.to(device)  # TE(16,24,295)--->(16,24,1,295)
        TE = self.FC_te(TE)  # TE(16,24,1,295)--->(16,24,1,64)
        del dayofweek, timeofday
        return SE + TE


class Attention(nn.Module):
    def __init__(self, QL, KL, VL, d, K, batch_size):
        super(Attention, self).__init__()


class spatialAttention(nn.Module):
    """
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, K, d, bn_decay):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_qkv = FC(input_dims=2 * D, units=D, activations=F.relu,
                         bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)
        self.modlist = nn.ModuleList([
            nn.Linear(64 * 1, 64),
            nn.Linear(64 * 2, 64),
            nn.Linear(64 * 3, 64),
            nn.Linear(64 * 4, 64),
            nn.Linear(64 * 5, 64),
            nn.Linear(64 * 6, 64),
            nn.Linear(64 * 7, 64),
            nn.Linear(64 * 8, 64),
            nn.Linear(64 * 9, 64),
            nn.Linear(64 * 10, 64),
            nn.Linear(64 * 11, 64),
            nn.Linear(64 * 12, 64),
        ])

    def forward(self, X, STE):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        XL = list(torch.split(X, 1, dim=1))  # XL存储了12个时间片数据
        Q = []
        K = []
        V = []
        for i in range(12):
            Q.append(self.FC_qkv(XL[i]))
            K.append(self.FC_qkv(XL[i]))
            V.append(self.FC_qkv(XL[i]))  # (16, 1, 325, 64)

            Q[i] = torch.cat(torch.split(Q[i], self.K, dim=-1),
                             dim=0)  # 将(16,1,325,64)分成8块torch.Size([16, 1, 325, 8])再在第一维拼起来得到（[128, 1, 325, 8]）
            K[i] = torch.cat(torch.split(K[i], self.K, dim=-1), dim=0)
            V[i] = torch.cat(torch.split(V[i], self.K, dim=-1), dim=0)
        Xt = []
        step = 0
        for m in self.modlist:
            T = []
            for j in range(step + 1):
                Qt = Q[step]
                Kt = K[j]
                Vt = V[j]
                T.append(attention_f(Qt, Kt, Vt, self.d, self.K, batch_size))
            T = torch.cat(T, dim=-1)
            T = m(T)
            step += 1
            Xt.append(T)
            del T
        X = torch.cat(Xt, dim=1)
        X = self.FC(X)

        """query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)"""

        # [K * batch_size, num_step, num_vertex, d]
        """query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)"""
        # [K * batch_size, num_step, num_vertex, num_vertex]
        """attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)"""
        # [batch_size, num_step, num_vertex, D]
        """X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size         将X(128, 12, 325, 8)分成8块torch.Size([16, 12, 325, 8])再在最后一维拼起来得到X（[16, 12, 325, 64]）
        X = self.FC(X)"""
        del Q, K, V
        return X


class temporalAttention(nn.Module):
    """
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1),
                          dim=0)  # 分成了8块torch.Size([16, 12, 325, 8])又在第一维上拼接起来得到torch.Size([128, 12, 325, 8])
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    """
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)


class transformAttention(nn.Module):
    """
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    """

    def __init__(self, K, d, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class TBSCN(nn.Module):
    """
    GMAN
        X：       [batch_size, num_his, num_vertx]
        TE：      [batch_size, num_his + num_pred, 2] (time-of-day, day-of-week)
        SE：      [num_vertex, K * d]
        num_his： number of history steps
        num_pred：number of prediction steps
        T：       one day is divided into T steps
        L：       number of STAtt blocks in the encoder/decoder
        K：       number of attention heads
        d：       dimension of each attention head outputs
        return：  [batch_size, num_pred, num_vertex]
    """

    def __init__(self, SE, args, bn_decay):
        super(TBSCN, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        self.num_his = args.num_his
        self.SE = SE
        self.STEmbedding = STEmbedding(D, bn_decay)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, X, TE, device):

        # input
        X = X.to(device)
        TE = TE.to(device)
        X = torch.unsqueeze(X, -1)  # 添加最后一个维度变为(16,12,325,1)维
        X = self.FC_1(X)  # (16,12,325,64)
        # STE
        SE = self.SE
        SE = SE.to(device)
        STE = self.STEmbedding(SE, TE)  # STE(16,24,325,64)
        STE_his = STE[:, :self.num_his]  # STE_his(16,12,325,64)
        STE_pred = STE[:, self.num_his:]  # STE_pred(16,12,325,64)
        # encoder                                                                        #  首先修改部分
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        # transAtt
        X = self.transformAttention(X, STE_his, STE_pred)
        # decoder
        for net in self.STAttBlock_2:  # 其次修改部分
            X = net(X, STE_pred)
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)
