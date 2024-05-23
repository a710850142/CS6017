import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt

# 计算均值函数
def calc_mean(data):
    return sum(data) / len(data)

# 计算标准差函数
def calc_std(data):
    mean = calc_mean(data)
    return sqrt(sum((x - mean)**2 for x in data) / len(data))

# 从标准正态分布中抽取1000个样本
samples = norm.rvs(size=1000)

# 计算样本均值和标准差
print(f"Sample Mean (Custom Function): {calc_mean(samples):.2f}, Standard Deviation (Custom Function): {calc_std(samples):.2f}")
print(f"Sample Mean (Numpy): {np.mean(samples):.2f}, Standard Deviation (Numpy): {np.std(samples):.2f}")

# 绘制样本直方图
plt.figure(figsize=(8, 4))
plt.hist(samples, bins=30, edgecolor='black')
plt.title("Sample Histogram")
plt.show()

# 读取CSV数据
df = pd.read_csv('2021-PM2.5.csv')

# 将日期列转换为datetime类型
df['Date'] = pd.to_datetime(df['Date'])

# 选择 'BV-MC' 监测站的数据
station_data = df[['Date', 'BV-MC']]

# 绘制全年PM2.5水平变化图
plt.figure(figsize=(12, 4))
plt.plot(station_data['Date'], station_data['BV-MC'])
plt.title(f"BV-MC Station PM2.5 Annual Trend")
plt.xticks(rotation=45)
plt.show()

# 按月分组并计算均值
monthly_avg = station_data.groupby(pd.Grouper(key='Date', freq='ME')).mean()

# 绘制月均值柱状图
plt.figure(figsize=(10, 4))
monthly_avg.plot.bar(rot=0)
plt.title(f"BV-MC Station PM2.5 Monthly Average")
plt.ylabel("PM2.5")
plt.show()

# 提取小时信息
station_data = station_data.copy()
station_data.loc[:, 'hour'] = station_data['Date'].dt.hour

# 按小时分组并计算均值
hourly_avg = station_data.groupby('hour').mean()

# 绘制小时均值柱状图
plt.figure(figsize=(10, 4))
hourly_avg.plot.bar(rot=0)
plt.title(f"BV-MC Station PM2.5 Hourly Average")
plt.xlabel("Hour")
plt.ylabel("PM2.5")
plt.show()

# 绘制月度箱线图
plt.figure(figsize=(10, 4))
station_data.boxplot(column=['BV-MC'], by=station_data['Date'].dt.month, xlabel='Month')
plt.title(f"BV-MC Station PM2.5 Monthly Boxplot")
plt.xlabel("Month")
plt.suptitle("")
plt.show()

# 绘制小时箱线图
plt.figure(figsize=(12, 4))
station_data.boxplot(column=['BV-MC'], by='hour', xlabel='Hour')
plt.title(f"BV-MC Station PM2.5 Hourly Boxplot")
plt.xlabel("Hour")
plt.suptitle("")
plt.xticks(rotation=45)
plt.show()
