import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from step123 import X_tensor


# 定义自编码器模型
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 添加噪声到图像
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn(*images.shape)
    return torch.clamp(noisy_images, 0., 1.)


# 训练自编码器
def train_autoencoder(model, train_loader, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            noisy_img = add_noise(img)

            output = model(noisy_img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 可视化结果
def visualize_denoising(model, test_images):
    model.eval()
    with torch.no_grad():
        noisy_images = add_noise(test_images)
        denoised_images = model(noisy_images)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(test_images):
                axes[i, j].imshow(test_images[idx].squeeze().cpu(), cmap='gray')
                axes[i, j].set_title('Original')
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('original_images.png')
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(noisy_images):
                axes[i, j].imshow(noisy_images[idx].squeeze().cpu(), cmap='gray')
                axes[i, j].set_title('Noisy')
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('noisy_images.png')
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(denoised_images):
                axes[i, j].imshow(denoised_images[idx].squeeze().cpu(), cmap='gray')
                axes[i, j].set_title('Denoised')
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('denoised_images.png')
    plt.close()


# 主要步骤
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用之前的 X_tensor
autoencoder = DenoisingAutoencoder().to(device)
autoencoder_train_loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=32, shuffle=True)

# 训练自编码器
train_autoencoder(autoencoder, autoencoder_train_loader)

# 可视化结果
test_images = X_tensor[:9].to(device)  # 选择9张图像进行可视化
visualize_denoising(autoencoder, test_images)

print("Denoising autoencoder training and visualization complete!")