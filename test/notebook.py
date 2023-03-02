# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.9)
"""1、optimizer要更改学习率的优化器，
   2、每训练step_size个epoch，更新一次参数，
   3、更新lr的乘法因子
   4、last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始。"""
# model.eval()
# with torch.no_grad():
"""
model.eval() 作用等同于 self.train(False)
简而言之，就是评估模式。而非训练模式。
在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
"""

# str.split(str="",num=string.count(str))[n]
"""
split()：语法：str.split(str="",num=string.count(str))[n]
拆分字符串。通过制定分隔符将字符串进行切片，并返回分割后的字符串列表[list]
参数：str：分隔符，默认为空格，但不能为空("")
num: 表示分割次数。如果指定num，则分割成n+1个子字符串，并可将每个字符串赋给新的变量
[n]: 选取第n个分片，即第n个字符串，从0开始算。
"""
# 1.时间序列 TimeSeries：以Datetimeindex为index的Series，就是时间序列。
# 2.pandas.date_range():直接生成Datetimeindex
# 两种生成方式：1.start+end ; 2.start / end+period
# 默认频率：天
