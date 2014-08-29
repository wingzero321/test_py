# test.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s=pd.Series([1,3,4,np.nan,6,8])

print s

dates=pd.date_range('20130101',periods=6)
print dates

df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))

print df

df2=pd.DataFrame({'A':1.,
	'B':pd.Timestamp('20130102'),
	'C':pd.Series(1,index=list(range(4)),dtype='float32'),
	'D':np.array([3]*4,dtype='int32'),
	'E':'foo'
		})

print df2
print df2.dtypes

# viewing DataFrame

print df.head(2)
print df.tail(2)

print df.index

print df.columns

print df.values

# transpose
print df.T

# sorting

print df.sort(columns='B')

# Getting var

print df['A'],df[0:3],df['20130102':'20130104']

print df.loc[dates[0]]

print df.loc[:,['A','B']]

print df.loc['20130102':'20130104',['A','B']]

# Boolean indexing

print df[df.A>0]

df2=df.copy()

df2['E']=['one', 'one','two','three','four','three']

print df2

print df2[df2['E'].isin(['two','four'])]

# Setting
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130102',periods=6))
df['F'] = s1
df.at[dates[0],'A'] = 0
df.iat[0,1] = 0
df.loc[:,'D'] = np.array([5] * len(df))
print df

# missing Data
df1 = df.reindex(index=dates[0:4],columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
print df1

print df1.dropna(how='any')

print df1.fillna(value=5)

print pd.isnull(df1)

# stats

print df.mean()

print df.mean(1)

s = pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2)

print s

print df.sub(s,axis='index')

print df.apply(np.cumsum)

print df.apply(lambda x: x.max() - x.min())

df = pd.DataFrame(np.random.randn(10, 4))

print df

pieces = [df[:3], df[3:7], df[7:]]

print pieces

print 'xxxxxxxxx\n',pd.concat(pieces)

# join

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
merge_data=pd.merge(left, right, on='key')

print merge_data

# append
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])

s = df.iloc[3]

df.append(s, ignore_index=True)

print df

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                    'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                    'C' : np.random.randn(8),
                    'D' : np.random.randn(8)})

print df

# summary

print df.groupby('A').sum()

print df.groupby(['A','B']).sum()

# pivot tables

df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                       'B' : ['A', 'B', 'C'] * 4,
                       'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                       'D' : np.random.randn(12),
                       'E' : np.random.randn(12)})

print pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

# time Series
rng = pd.date_range('1/1/2012', periods=100, freq='S')

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

ts.resample('5Min', how='sum')

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
# plt.show()

# getting Data in / out

df.to_csv('foo.csv')

df_1=pd.read_csv('foo.csv')

print df_1

df.to_excel('foo.xlsx', sheet_name='Sheet1')

print pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

