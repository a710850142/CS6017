import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 加载CSV数据
uniform_data = pd.read_csv('uniform_output.csv')
gaussian_data = pd.read_csv('gaussian_output.csv')

# 定义数据结构列表和颜色映射
structures = ['dumbKNN', 'bucketKNN', 'quadTree', 'kdTree']
color_map = {'dumbKNN': 'red', 'bucketKNN': 'green', 'quadTree': 'blue', 'kdTree': 'orange'}

# 创建图形
fig, axs = plt.subplots(3, 2, figsize=(20, 20))

# 对每个数据集进行分析
for i, data in enumerate([uniform_data, gaussian_data]):

    # 固定k和N,变化D
    for struct in structures:
        sub_data = data[(data['data_structure'] == struct) & (data['k'] == 10) & (data['N'] == 10000)]
        if not sub_data.empty:
            axs[0, i].plot(sub_data['D'], sub_data['time'], label=struct, color=color_map[struct])
    axs[0, i].set_xlabel('D')
    axs[0, i].set_ylabel('Time (µs)')
    axs[0, i].set_title(f'Fixed k=10, N=10000 ({["Uniform", "Gaussian"][i]})')
    axs[0, i].legend()

    # 固定D和N,变化k
    for struct in structures:
        sub_data = data[(data['data_structure'] == struct) & (data['D'] == 3) & (data['N'] == 10000)]
        if not sub_data.empty:
            axs[1, i].plot(sub_data['k'], sub_data['time'], label=struct, color=color_map[struct])
    axs[1, i].set_xlabel('k')
    axs[1, i].set_ylabel('Time (µs)')
    axs[1, i].set_title(f'Fixed D=3, N=10000 ({["Uniform", "Gaussian"][i]})')
    axs[1, i].legend()

    # 固定D和k,变化N
    for struct in structures:
        sub_data = data[(data['data_structure'] == struct) & (data['D'] == 3) & (data['k'] == 10)]
        if not sub_data.empty:
            axs[2, i].plot(sub_data['N'], sub_data['time'], label=struct, color=color_map[struct])
    axs[2, i].set_xlabel('N')
    axs[2, i].set_ylabel('Time (µs)')
    axs[2, i].set_title(f'Fixed D=3, k=10 ({["Uniform", "Gaussian"][i]})')
    axs[2, i].legend()

plt.tight_layout()
plt.show()

# 回归分析
print("Regression Analysis:")
for struct in structures:
    print(f"\nData Structure: {struct}")

    # D的影响
    sub_data = uniform_data[
        (uniform_data['data_structure'] == struct) & (uniform_data['k'] == 10) & (uniform_data['N'] == 10000)]
    if not sub_data.empty:
        X = sub_data['D'].values.reshape(-1, 1)
        y = sub_data['time'].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        print(f"Time = {reg.coef_[0][0]:.2f} * D + {reg.intercept_[0]:.2f}")
    else:
        print("Insufficient data for regression analysis on D.")

    # k的影响
    sub_data = uniform_data[
        (uniform_data['data_structure'] == struct) & (uniform_data['D'] == 3) & (uniform_data['N'] == 10000)]
    if not sub_data.empty:
        X = sub_data['k'].values.reshape(-1, 1)
        y = sub_data['time'].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        print(f"Time = {reg.coef_[0][0]:.2f} * k + {reg.intercept_[0]:.2f}")
    else:
        print("Insufficient data for regression analysis on k.")

    # N的影响
    sub_data = uniform_data[(uniform_data['data_structure'] == struct) & (uniform_data['D'] == 3)]
    if not sub_data.empty:
        X = sub_data['N'].values.reshape(-1, 1)
        y = sub_data['time'].values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        reg = LinearRegression().fit(X_poly, y)
        print(f"Time = {reg.coef_[0][1]:.2e} * N^2 + {reg.coef_[0][0]:.2e} * N")
    else:
        print("Insufficient data for regression analysis on N.")

# 热力图 (k和N的影响)
for i, data in enumerate([uniform_data, gaussian_data]):
    sub_data = data[(data['D'] == 2)]
    if not sub_data.empty:
        pivot_data = sub_data.pivot_table(index='k', columns='N', values='time', aggfunc='mean')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlGnBu')
        plt.xlabel('N')
        plt.ylabel('k')
        plt.title(f'Heatmap of Time for Different k and N ({["Uniform", "Gaussian"][i]}, D=2)')
        plt.show()
    else:
        print(f"Insufficient data for heatmap ({['Uniform', 'Gaussian'][i]}, D=2).")