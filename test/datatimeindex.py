import pandas as pd
import numpy as np

t1 = pd.DatetimeIndex(['2017/8/1', '2018/8/2', '2018/8/3', '2018/8/4/', '2018/8/5'])
print(t1, type(t1))
"""
DatetimeIndex(['2017-08-01', '2018-08-02', '2018-08-03', '2018-08-04',
               '2018-08-05'],
               dtype='datetime64[ns]', freq=None) <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
"""
# 先创建一个Datetimeindex,再创建一个相同长度的Series，将Datetimeindex作为index
st = pd.Series(np.random.rand(len(t1)), index=t1)
print(st, type(st))
print(st.index)
