import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 加载数据
df = pd.read_csv('hackernews_stories.csv')

# 实现排名公式
def rank_formula(points, age, comments):
    gravity = 1.8
    time_decay = 0.95
    return (points - 1) / (age + 2) ** gravity + comments / (age + 2) ** time_decay

# 计算 "真实" 排名
df['true_rank'] = rank_formula(df['Points'], df['Age(hours)'], df['Comments'])

# 准备特征和目标变量
X = df[['Points', 'Age(hours)', 'Comments']]
y = df['true_rank']

# 执行最小二乘回归
model = LinearRegression()
model.fit(X, y)

# 比较估计的系数与实际系数
estimated_coefficients = model.coef_
actual_coefficients = [1, -1.8, 1]  # 根据排名公式的实际系数
print("Estimated Coefficients:", estimated_coefficients)
print("Actual Coefficients:", actual_coefficients)

# 计算误差度量或差异
coefficient_errors = estimated_coefficients - actual_coefficients
print("Coefficient Errors:", coefficient_errors)

# 计算均方误差 (MSE)
y_pred = model.predict(X)
mse = np.mean((y_pred - y) ** 2)
print("Mean Squared Error:", mse)

# 解释结果
print("\nResults Interpretation:")
print("- The estimated coefficients from the linear regression are:")
print("  Points: {:.2f}, Age(hours): {:.2f}, Comments: {:.2f}".format(*estimated_coefficients))
print("- The actual coefficients used in the ranking formula are:")
print("  Points: {}, Age(hours): {}, Comments: {}".format(*actual_coefficients))
print("- The coefficient errors (estimated - actual) are:")
print("  Points: {:.2f}, Age(hours): {:.2f}, Comments: {:.2f}".format(*coefficient_errors))
print("- The Mean Squared Error (MSE) between the predicted and true ranks is: {:.2f}".format(mse))
print("- The linear regression model tries to estimate the coefficients of the ranking formula.")
print("- The closer the estimated coefficients are to the actual coefficients, the better the model.")
print("- A lower MSE indicates that the predicted ranks are closer to the true ranks.")
print("- The results suggest that the linear regression model can approximate the ranking formula,")
print("  but there are still some differences between the estimated and actual coefficients.")
print("- Factors such as data quality, sample size, and the complexity of the ranking formula")
print("  can affect the accuracy of the coefficient estimation.")