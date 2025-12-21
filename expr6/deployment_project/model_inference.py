# -------------------- model_inference.py --------------------
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 假设类别标签 (需要和训练时的顺序一致)
CLASS_NAMES = ['Covid', 'Normal', 'Viral Pneumonia']
# 强制使用 CPU 进行部署推理，以避免服务器没有 GPU 引起的错误
DEVICE = torch.device('cpu')


# --- 1. 定义模型结构 (必须与训练时保持一致) ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # 解码器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=len(CLASS_NAMES))  # 3 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- 2. 模型加载和预处理函数 ---
def load_models(ae_path, cnn_path):
    """加载自编码器和CNN模型权重。"""
    try:
        # 加载 Autoencoder
        ae_model = Autoencoder().to(DEVICE)
        ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
        ae_model.eval()

        # 加载 CNN
        cnn_model = CNN().to(DEVICE)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        cnn_model.eval()

        print("Models loaded successfully.")
        return ae_model, cnn_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None


def preprocess_image(image_path):
    """加载图片并执行与训练时一致的预处理。"""

    # 图像预处理定义 (必须与训练时一致)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    try:
        image = Image.open(image_path)
        # 1. 预处理
        tensor_image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
        return tensor_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# --- 3. 核心预测函数 ---
def predict_image(image_path, ae_model, cnn_model):
    """加载、去噪、分类图片并返回结果。"""

    # 1. 预处理图片 (含ToTensor)
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return "Prediction Failed (Preprocess Error)", 0.0

    with torch.no_grad():
        # 2. 去噪 (通过自编码器)
        denoised_tensor = ae_model(input_tensor)

        # 3. 分类 (通过 CNN)
        output = cnn_model(denoised_tensor)

        # 4. 计算概率和预测类别
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

        predicted_class = CLASS_NAMES[predicted_index.item()]
        confidence_score = confidence.item()

    return predicted_class, confidence_score


# 仅在需要单独测试时执行加载
if __name__ == '__main__':
    ae, cnn = load_models('../autoencoder_model.pth', '../cnn_model.pth')
    if ae and cnn:
        # 替换为实际图片路径进行测试
        test_file = './test_image.jpeg'
        if os.path.exists(test_file):
            result, confidence = predict_image(test_file, ae, cnn)
            print(f"Prediction for {test_file}: {result} ({confidence:.2f})")
        else:
            print("Test image not found.")