import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import itertools

# 读取数据
df = pd.read_csv('hackernews_stories.csv')

# 数据探索
print(df.describe())

# 选择只包含数值列的子集
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(df[numeric_columns].corr())

# 计算标题长度
df['Title_Length'] = df['Title'].apply(len)

# 准备特征和目标变量
X = df[['Points', 'Comments', 'Title_Length']]
y = df['Rank']

# 线性回归
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)
print(f"Linear Regression Coefficients: {lin_reg.coef_}")
print(f"Linear Regression R-squared: {r2_score(y, y_pred_lin)}")

# 多项式回归
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
print(f"Polynomial Regression R-squared: {r2_score(y, y_pred_poly)}")

# 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
y_pred_ridge = ridge.predict(X)
print(f"Ridge Regression Coefficients: {ridge.coef_}")
print(f"Ridge Regression R-squared: {r2_score(y, y_pred_ridge)}")

# 绘制实际rank与预测rank的散点图
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(y, y_pred_lin)
plt.xlabel('Actual Rank')
plt.ylabel('Predicted Rank')
plt.title('Linear Regression')

plt.subplot(2, 2, 2)
plt.scatter(y, y_pred_poly)
plt.xlabel('Actual Rank')
plt.ylabel('Predicted Rank')
plt.title('Polynomial Regression')

plt.subplot(2, 2, 3)
plt.scatter(y, y_pred_ridge)
plt.xlabel('Actual Rank')
plt.ylabel('Predicted Rank')
plt.title('Ridge Regression')

plt.tight_layout()
plt.show()

# 尝试不同的特征组合
features = ['Points', 'Comments', 'Age(hours)', 'Title_Length']
for i in range(1, len(features)+1):
    for combo in itertools.combinations(features, i):
        X = df[list(combo)]
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        y_pred = lin_reg.predict(X)
        print(f"Features: {combo}, R-squared: {r2_score(y, y_pred)}")