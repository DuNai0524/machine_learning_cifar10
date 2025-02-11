import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
import data
from torch import nn
from model.model import ResNet18

from utils.Timer import Timer
from utils.Accumulator import Accumulator
import utils.GetDevice as getD

import matplotlib

# 文件路径
data_dir = './data'

# 模型初始化,个人认为非常重要，特别是容易梯度爆炸和消失的情况中。或许在层中添加一个BN层也非常好
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    num_classes = 10
    net = ResNet18(num_classes)
    return net

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def main():
    # 划分数据
    batch_size = 128
    valid_ratio = 0.1
    # data.reorg_cifar10_data(data_dir, valid_ratio)
    print("=====开始训练=====")
    # 图像增广
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(40),
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    """测试数据只对图片通道标准化"""
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    # 读取数据
    train_ds, train_valid_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_train) for folder in ['train', 'train_valid']]
    
    valid_ds, test_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_test) for folder in ['valid', 'test']]
    
    train_iter, train_valid_iter = [
        torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]
    
    # drop_last 表示如果是最后一个 batch 不够就丢掉
    valid_iter = torch.utils.data.DataLoader(
        valid_ds, batch_size, shuffle=False, drop_last=True)
    
    test_iter = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle=False, drop_last=False)
    
    devices, num_epochs, lr, wd = getD.try_all_gpus(), 50,  0.001, 5e-4
    lr_period, lr_decay, net = 10, 0.1, get_net()
    # net.apply(init_weights)
    train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)
    
    preds = []
    net.eval()
    for X, _ in test_iter:
        y_hat = net(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    
    # 生成文件
    sorted_ids = list(range(1, test_ds.__len__() + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
    df.to_csv('submission.csv', index=False)
    

def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


# 定义训练代码
import matplotlib.pyplot as plt

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay) # 学习率下降
    num_batches, timer = len(train_iter), Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    # 如果使用 reduction="none"不收敛，使用默认的elementwise_mean
    loss = nn.CrossEntropyLoss()
    
    # 初始化用于存储每个 epoch 的 train_loss 和 train_acc 的列表
    train_losses = []
    train_accs = []
    valid_accs = []
    
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
        
        scheduler.step()
        
        if valid_iter is not None:
            valid_acc = evaluate_accuracy_gpu(net, valid_iter)
            valid_accs.append(valid_acc)
        # 计算并存储每个 epoch 的 train_loss 和 train_acc
        train_loss = metric[0]*100 / metric[2]
        train_acc = metric[1] / metric[2]
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, valid Acc: {valid_acc:.3f}')

    measures = (f'train loss {metric[0]*100 / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    
    # 绘制 train_loss、train_acc 和 valid_acc 的图像并保存
    plt.figure(figsize=(10, 5))
    
    # 绘制 Training Loss
    plt.plot(train_losses, label='Train Loss')
    
    # 绘制 Training Accuracy
    plt.plot(train_accs, label='Train Acc')
    
    # 绘制 Validation Accuracy
    plt.plot(valid_accs, label='Valid Acc')
    
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('result_adam_50.png')  # 保存图像为 PNG 文件


if __name__ == '__main__':
    main()

