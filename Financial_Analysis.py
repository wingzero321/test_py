# Financial_Analysis.py

import datetime

import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
# import scipy

aapl = pd.io.data.get_data_yahoo('AAPL', 
                                 start=datetime.datetime(2006, 10, 1), 
                                 end=datetime.datetime(2014, 1, 1))
print aapl.head()

aapl.to_csv('aapl_ohlc.csv')
df = pd.read_csv('aapl_ohlc.csv', index_col='Date', parse_dates=True)

ts=df['Close'][-10:]

print ts

date = ts.index[5]

ts[date]

df['diff'] = df.Open - df.Close

print df.head()

close_px = df['Adj Close']
print close_px.head(40)

# calculate moving average
mavg = pd.rolling_mean(close_px, 40)
print mavg.head(40)

# returns
rets=close_px/close_px.shift(1)-1

# close_px.plot(label='AAPL')
# mavg.plot(label='mavg')
# plt.legend()
# plt.show()

df = pd.io.data.get_data_yahoo(['AAPL', 'GE', 'IBM', 'KO', 'MSFT', 'PEP'], 
                               start=datetime.datetime(2010, 1, 1), 
                               end=datetime.datetime(2014, 1, 1))['Adj Close']
print df.head()

rets = df.pct_change()

print rets

# plt.scatter(rets.AAPL, rets.IBM)
# plt.xlabel('Returns AAPL')
# plt.ylabel('Returns IBM')


# pd.scatter_matrix(rets, diagonal='kde', figsize=(9, 9))

# plt.show()

corr=rets.corr()

print corr

# plt.imshow(corr, cmap='hot', interpolation='none')
# plt.colorbar()
# plt.xticks(range(len(corr)), corr.columns)
# plt.yticks(range(len(corr)), corr.columns);

# plt.show()

plt.scatter(rets.mean(), rets.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()



