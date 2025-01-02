import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torchvision
import pandas as pd
import cv2
import numpy as np

class ImageClassifier:
    def __init__(self, model_path, labels_csv, device=None):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.device = device if device else self.get_device()
        self.net = self.get_net()
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()

        df = pd.read_csv(labels_csv)
        self.classes = df['label'].unique().tolist()

    def get_device(self):
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        if not devices:
            devices = [torch.device('cpu')]
        return devices[0]

    def get_net(self):
        finetune_net = nn.Sequential()
        finetune_net.features = torchvision.models.resnet34()
        finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 120))
        finetune_net = finetune_net.to(self.device)
        for param in finetune_net.features.parameters():
            param.requires_grad = False
        return finetune_net

    def predict(self, image):
        if isinstance(image, np.ndarray):
            # 将 numpy 数组转换为 PIL 图像
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.net(image)
            _, predicted = torch.max(output, 1)
            return self.classes[predicted.item()]

# 使用示例
if __name__ == "__main__":
    classifier = ImageClassifier(model_path='model34.pth', labels_csv='labels.csv')
    image_path = './test_img.jpg'  # 替换为实际图片路径
    predicted_class = classifier.predict(image_path)
    print(f'预测的类别是: {predicted_class}')