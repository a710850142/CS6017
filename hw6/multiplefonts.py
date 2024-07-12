import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# 定义 CharacterCNN 类
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


def load_and_preprocess_data(file_path, unicode_to_index=None):
    df = pd.read_csv(file_path)
    Xs, Ys = transform_data(df)
    if unicode_to_index is None:
        unicode_to_index, _ = create_label_dicts(Ys.flatten())
    X_tensor = torch.FloatTensor(Xs).unsqueeze(1)
    y_tensor = torch.LongTensor([unicode_to_index.get(y[0], -1) for y in Ys])
    return X_tensor, y_tensor, unicode_to_index


def train_and_evaluate(model, train_loader, test_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')

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
    return accuracy


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载所有数据以创建完整的unicode_to_index字典
df_agency = pd.read_csv('AGENCY.csv')
df_baiti = pd.read_csv('BAITI.csv')
df_arial = pd.read_csv('ARIAL.csv')

all_labels = np.concatenate([df_agency['m_label'].values, df_baiti['m_label'].values, df_arial['m_label'].values])
unicode_to_index, _ = create_label_dicts(all_labels)

# 现在使用这个完整的字典加载数据
X_agency, y_agency, _ = load_and_preprocess_data('AGENCY.csv', unicode_to_index)
X_baiti, y_baiti, _ = load_and_preprocess_data('BAITI.csv', unicode_to_index)
X_arial, y_arial, _ = load_and_preprocess_data('ARIAL.csv', unicode_to_index)

# 单字体训练（AGENCY）
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_agency, y_agency, test_size=0.2,
                                                                                random_state=42)
train_loader_single = DataLoader(TensorDataset(X_train_single, y_train_single), batch_size=32, shuffle=True)
test_loader_single = DataLoader(TensorDataset(X_test_single, y_test_single), batch_size=32, shuffle=False)

model_single = CharacterCNN(num_classes=len(unicode_to_index)).to(device)
print("Training single font model (AGENCY)...")
accuracy_single = train_and_evaluate(model_single, train_loader_single, test_loader_single)

# 多字体训练（AGENCY + BAITI）
X_multi = torch.cat((X_agency, X_baiti), 0)
y_multi = torch.cat((y_agency, y_baiti), 0)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2,
                                                                            random_state=42)
train_loader_multi = DataLoader(TensorDataset(X_train_multi, y_train_multi), batch_size=32, shuffle=True)
test_loader_multi = DataLoader(TensorDataset(X_test_multi, y_test_multi), batch_size=32, shuffle=False)

model_multi = CharacterCNN(num_classes=len(unicode_to_index)).to(device)
print("Training multi-font model (AGENCY + BAITI)...")
accuracy_multi = train_and_evaluate(model_multi, train_loader_multi, test_loader_multi)

# 在未见过的字体上测试（ARIAL）
test_loader_unseen = DataLoader(TensorDataset(X_arial, y_arial), batch_size=32, shuffle=False)

print("Testing on unseen font (ARIAL)...")
accuracy_unseen_single = train_and_evaluate(model_single, train_loader_single, test_loader_unseen, num_epochs=0)
accuracy_unseen_multi = train_and_evaluate(model_multi, train_loader_multi, test_loader_unseen, num_epochs=0)

print(f"Single font (AGENCY) accuracy: {accuracy_single:.2f}%")
print(f"Multi font (AGENCY + BAITI) accuracy: {accuracy_multi:.2f}%")
print(f"Unseen font (ARIAL) accuracy - Single font model: {accuracy_unseen_single:.2f}%")
print(f"Unseen font (ARIAL) accuracy - Multi font model: {accuracy_unseen_multi:.2f}%")