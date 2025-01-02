#ImageNet_Dogs.py
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

# 获取当前ImageNet_Dogs.py文件的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 下载和整理数据集
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip', '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')
demo = True
data_dir = d2l.download_extract('dog_tiny') if demo else os.path.join('..', 'data', 'dog-breed-identification')
print("数据集路径：", data_dir, "数据集内容：", os.listdir(data_dir))

def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# 图像增广
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
                                for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

# 保存标签为 .csv 文件
output_csv = os.path.join(script_dir, 'labels.csv')
data = []
for class_name in train_valid_ds.classes:
    class_dir = os.path.join(data_dir, 'train_valid_test', 'train_valid', class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            data.append([img_path, class_name])

df = pd.DataFrame(data, columns=['image_path', 'label'])
df.to_csv(output_csv, index=False)
print(f'标签已保存到 {output_csv}')

# 定义模型
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet50(pretrained=True)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

# 计算损失
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')

# 训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])  # 模型并行化
    trainer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)  # 定义优化器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)  # 定义学习率调度器
    num_batches, timer = len(train_iter), d2l.Timer()  # 初始化计时器
    train_losses, valid_losses = [], []  # 初始化训练损失和验证损失
    for epoch in range(num_epochs):  # 训练循环周期
        metric = d2l.Accumulator(2)  # 初始化累加器，用于累加损失和样本数。
        with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:  # 进度条
            for i, (features, labels) in enumerate(train_iter):  # 遍历每个批次
                timer.start()
                features, labels = features.to(devices[0]), labels.to(devices[0])
                trainer.zero_grad()  # 梯度清零
                output = net(features)  # 前向传播
                l = loss(output, labels).sum()
                l.backward()  # 反向传播
                trainer.step()  # 更新参数
                metric.add(l, labels.shape[0])  # 累加损失和样本数
                timer.stop()
                pbar.update(1)
        train_loss = metric[0] / metric[1]  # 记录当前epoch的训练损失
        train_losses.append(train_loss)  # 将训练损失添加到列表中

        if valid_iter is not None:  # 如果有验证集，计算验证损失。
            valid_loss = evaluate_loss(valid_iter, net, devices)
            valid_losses.append(valid_loss.detach().cpu())
        scheduler.step()  # 更新学习率

    print(f'train loss {train_loss:.3f}')
    if valid_iter is not None:  # 记录验证损失
        print(f'valid loss {valid_loss:.3f}')
    print(f'\n{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

# 训练和验证模型
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)
# 保存模型
model_path = os.path.join(script_dir, 'model.pth')
torch.save(net.state_dict(), model_path)
print(f'模型已保存到 {model_path}')

import numpy as np

# 加载并使用模型
net = get_net(devices)
net.load_state_dict(torch.load('model34.pth'))
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 获取预测结果
preds = []
for data, label in test_iter:  # 遍历所有测试集样本
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())

# 获取测试集图像文件名
ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
ids = ids[:len(preds)]  # 确保 ids 和 preds 的长度相同

# 定义类别
classes = train_valid_ds.classes

# 计算平均置信度和各个置信度区间所占比例
performance = [np.max(pred) for pred in preds]
mean_confidence = np.mean(performance)
above_90 = np.sum(np.array(performance) >= 0.9) / len(performance) * 100
below_50 = np.sum(np.array(performance) < 0.5) / len(performance) * 100

# 打印信息
print(f'平均置信度: {mean_confidence:.2f}')
print(f'90%以上所占比例: {above_90:.2f}%')
print(f'50%以下所占比例: {below_50:.2f}%')