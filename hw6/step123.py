import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Step 1: 数据处理

def transform_data(df):
    Ys = df['m_label'].values.reshape(-1, 1)
    pixel_columns = [col for col in df.columns if col.startswith('r') and 'c' in col]
    Xs = df[pixel_columns].values
    Xs = Xs.reshape(-1, 20, 20)
    Xs = Xs / 255.0
    return Xs, Ys


def create_label_dicts(labels):
    unicode_to_index = {unicode: idx for idx, unicode in enumerate(sorted(set(labels)))}
    index_to_unicode = {idx: unicode for unicode, idx in unicode_to_index.items()}
    return unicode_to_index, index_to_unicode


# 读取CSV文件
df = pd.read_csv('AGENCY.csv')

# 转换数据
Xs, Ys = transform_data(df)

# 创建标签字典
unicode_to_index, index_to_unicode = create_label_dicts(Ys.flatten())

# 将Y值转换为索引
Ys = np.array([unicode_to_index[y[0]] for y in Ys])

print("Xs shape:", Xs.shape)
print("Ys shape:", Ys.shape)
print("Number of unique labels:", len(unicode_to_index))

# Step 2: 模型训练

# 将数据转换为PyTorch张量
X_tensor = torch.FloatTensor(Xs).unsqueeze(1)  # 添加通道维度
y_tensor = torch.LongTensor(Ys)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义模型
class CharacterCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharacterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterCNN(num_classes=len(unicode_to_index)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')

print("Training complete!")

# 保存模型
torch.save(model.state_dict(), 'character_cnn_model.pth')


# Step 3: 评估和分析

# 1. 使用交叉验证评估模型
def cross_validate(X, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = CharacterCNN(num_classes=len(unicode_to_index)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):  # 训练10个epoch
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Fold {fold} Accuracy: {accuracy:.2f}%')

    print(f'Average Accuracy: {sum(accuracies) / len(accuracies):.2f}%')


cross_validate(X_tensor, y_tensor)


# 2. 分析错误分类的样本
def analyze_misclassifications(model, dataloader):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(batch_y)):
                if predicted[i] != batch_y[i]:
                    misclassified.append((batch_X[i], batch_y[i], predicted[i]))
    return misclassified


misclassified = analyze_misclassifications(model, test_loader)

# 显示一些错误分类的样本
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, (img, true_label, pred_label) in enumerate(misclassified[:9]):
    ax = axes[i // 3, i % 3]
    ax.imshow(img.cpu().squeeze(), cmap='gray')
    ax.set_title(f'True: {index_to_unicode[true_label.item()]}, Pred: {index_to_unicode[pred_label.item()]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('misclassified_samples.png')
plt.close()


# 3. 在不同字体上测试模型
def test_on_different_font(model, font_file):
    df_new = pd.read_csv(font_file)
    Xs_new, Ys_new = transform_data(df_new)
    X_new_tensor = torch.FloatTensor(Xs_new).unsqueeze(1)
    y_new_tensor = torch.LongTensor([unicode_to_index.get(y[0], -1) for y in Ys_new])

    new_dataset = TensorDataset(X_new_tensor, y_new_tensor)
    new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in new_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on new font: {accuracy:.2f}%')


# 另一个字体文件
test_on_different_font(model, 'ARIAL.csv')


# 4. 分析模型的不确定性
def analyze_uncertainty(model, dataloader):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            uncertainties.extend(max_probs.cpu().numpy())

    plt.figure(figsize=(10, 5))
    plt.hist(uncertainties, bins=50)
    plt.title('Distribution of Model Certainty')
    plt.xlabel('Max Probability')
    plt.ylabel('Count')
    plt.savefig('model_uncertainty.png')
    plt.close()


analyze_uncertainty(model, test_loader)

print("Evaluation and analysis complete!")