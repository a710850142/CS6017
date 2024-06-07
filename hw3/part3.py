import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 读取数据
df = pd.read_csv('hackernews_stories.csv')

# 创建一个新的二元特征 'is_front_page'，表示文章是否在首页（排名 <= 30）
df['is_front_page'] = (df['Rank'] <= 30).astype(int)

# 计算标题长度
df['Title_Length'] = df['Title'].apply(len)

# 准备特征和目标变量
X = df[['Points', 'Comments', 'Title_Length']]
y = df['is_front_page']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练逻辑回归模型
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = log_reg.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Front Page', 'Front Page'])
plt.yticks(tick_marks, ['Not Front Page', 'Front Page'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 绘制不同特征的logistic曲线
features = ['Points', 'Comments', 'Title_Length']

plt.figure(figsize=(12, 4))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    X_single = df[[feature]].values
    log_reg_single = LogisticRegression(random_state=42)
    log_reg_single.fit(X_single, y)

    x_min, x_max = X_single.min(), X_single.max()
    xx = np.linspace(x_min, x_max, 100)
    y_proba = log_reg_single.predict_proba(xx.reshape(-1, 1))[:, 1]

    plt.plot(xx, y_proba, color='blue', linewidth=2)
    plt.scatter(X_single, y, color='red', alpha=0.3)
    plt.xlabel(feature)
    plt.ylabel('Probability of Being on Front Page')
    plt.title(f'Logistic Curve for {feature}')

plt.tight_layout()
plt.show()