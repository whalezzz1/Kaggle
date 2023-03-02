import pandas as pd
import torch

df = pd.read_hdf('../data/pems-bay.h5')

time = pd.DatetimeIndex(df.index)  # df为数据文件pems-bay.h5，以Datetimeindex为index的Series，就是时间序列。
t1 = time.weekday  # 返回的0-6代表周一--到周日，每个片有一个数字代表周信息，每288个片为连续相同周几
dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))  # (52116,1)
t2 = time.hour  # 返回的0-23代表一天当中的24个小时，每个片有一个数字代表周信息，每12个片为连续相同小时
t3 = time.minute  # 返回的0，5，10，15，20，25...55，代表一个小时中的多少分钟，共12个不同信息
t4 = time.second
timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
            // 300  # 返回每天288个不同时间标记
timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))  # (52116,1)
print(timeofday)
time = torch.cat((dayofweek, timeofday), -1)  # 最后一维拼接 (52116,2)
print(time)
