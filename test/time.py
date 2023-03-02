import time

start_time = time.time()
print('当前时间：', start_time)
"""当前时间戳： 1669015475.669014   时间戳单位最适于做日期运算。但是1970年之前的日期就无法以此表示了。太遥远的日期也不行，UNIX和Windows只支持到2038年"""

localtime = time.localtime(time.time())
print("1、本地时间为 :", localtime)
"""从返回浮点数的时间戳方式向时间元组转换，只要将浮点数传递给如localtime之类的函数。"""

localtime = time.asctime(time.localtime(time.time()))
print("2、本地时间为 :", localtime)
"""根据需求选取各种格式，但是最简单的获取可读的时间模式的函数是asctime():"""

# 格式化成2016-03-20 11:45:39形式
print("3、", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 格式化成Sat Mar 28 22:24:24 2016形式
print("4、", time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

# 将格式字符串转换为时间戳
a = "Sat Mar 28 22:24:24 2016"
print("5、", time.mktime(time.strptime(a, "%a %b %d %H:%M:%S %Y")))

"""打印结果：
当前时间： 1669015942.7467375
1、本地时间为 : time.struct_time(tm_year=2022, tm_mon=11, tm_mday=21, tm_hour=15, tm_min=32, tm_sec=22, tm_wday=0, tm_yday=325, tm_isdst=0)
2、本地时间为 : Mon Nov 21 15:32:22 2022
3、 2022-11-21 15:32:22
4、 Mon Nov 21 15:32:22 2022
5、 1459175064.0
"""
