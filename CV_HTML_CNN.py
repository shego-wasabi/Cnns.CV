
import os
import pickle
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
from TensorNormalization import extract_features


def calculate_accuracy(labels_true, labels_pred):
    total_samples = len(labels_true)
    correct_samples = sum(1 for true, pred in zip(labels_true, labels_pred) if true == pred)
    accuracy = correct_samples / total_samples
    return accuracy

# 定义模型
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

# 定义数据集
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# 训练模型代码
def fit_module(input_size, num_classes, num_epochs, learning_rate, patience=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir_path = '/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/0520_labels'

    # 使用特征文件
    with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_feature_0520_CV.pkl', 'rb') as file:
        features_list = pickle.load(file)
    with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_label_0520_CV.pkl', 'rb') as file:
        labels_list = pickle.load(file)

    # 创建模型实例并移动到GPU
    model = Classifier(input_size, num_classes).to(device)
    # model.load_state_dict(torch.load("/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pth_files/05_20_152064_newlabel_CV_4.pth"))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 数据分割
    features_train, features_val, labels_train, labels_val = train_test_split(
        features_list, labels_list, test_size=0.2, random_state=42
    )

    feature_array = np.array(features_train)
    feature = torch.tensor(feature_array).to(device)

    # 早停逻辑
    best_val_loss = float('inf')
    no_improve_epoch = 0

    # 训练模型
    for epoch in range(num_epochs):
        # 创建一个从字符串到整数的映射（标签编码）
        label_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,'9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,'16': 16, '17': 17, '18':18, '19':19, '20':20}
        # label_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7,'9': 8, '10': 9}
        # 使用映射将字符串标签转换为整数标签
        label_train = [label_mapping[label_str] for label_str in labels_train]

        # 现在将整数标签列表转换为PyTorch张量
        labels_tensor = torch.tensor(label_train, dtype=torch.long)

        # 如果GPU可用，将标签张量移动到GPU上
        if torch.cuda.is_available():
            labels_tensor = labels_tensor.to(device)

        # 前向传播
        outputs = model(feature)
        # 计算损失
        loss = criterion(outputs, labels_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # 打印每个epoch的损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        # 验证过程
        model.eval()
        val_loss = 0
        val_accuracy = 0


        with torch.no_grad():

            label_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                             '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17,
                             '18': 18, '19': 19, '20': 20, '100': 100}

            # features, labels = features.to(device), labels.to(device)

            # label_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
            #                  '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19,
            #                  '20': 20}
            # # label_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7,'9': 8, '10': 9}
            # # 使用映射将字符串标签转换为整数标签
            # labels = [label_mapping[label_str] for label_str in label]
            #
            # # 现在将整数标签列表转换为PyTorch张量
            # labels_tensor = torch.tensor(labels, dtype=torch.long)
            feature_val_array = np.array(features_val)
            feature_val = torch.tensor(feature_val_array).to(device)


            label_val = [label_mapping[label_str] for label_str in labels_val]
            labels_val_tensor = torch.tensor(label_val, dtype=torch.long)
            if torch.cuda.is_available():
                labels_val_tensor = labels_val_tensor.to(device)

            outputs_val = model(feature_val)
            val_loss += criterion(outputs_val, labels_val_tensor)
            val_accuracy += calculate_accuracy(label_val, outputs_val.argmax(1))

        val_loss /= len(labels_val)
        val_accuracy /= len(labels_val)
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        # # 检查是否需要早停
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     no_improve_epoch = 0
        # else:
        #     no_improve_epoch += 1
        #     if no_improve_epoch >= patience:
        #         print("Early stopping triggered.")
        #         break


    torch.save(model.state_dict(), "/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pth_files/05_20_152064_newlabel_CV_5.pth")

def test_module(input_size, num_classes, model_path, test_data_path, labels_path, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Classifier(input_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载测试数据
    with open(test_data_path, 'rb') as file:
        test_features_list = pickle.load(file)
    with open(labels_path, 'rb') as file:
        test_labels_list = pickle.load(file)

    # 将测试特征转换为张量
    test_feature_array = np.array(test_features_list)
    test_feature = torch.tensor(test_feature_array).to(device)

    # 标签映射
    label_mapping = {str(i): i for i in range(num_classes)}  # 根据实际的类别数调整
    test_labels = [label_mapping[label_str] for label_str in test_labels_list]
    labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(test_feature)
        _, predicted = torch.max(outputs, 1)

    # 计算准确率
    correct = (predicted == labels_tensor).sum().item()
    accuracy = correct / len(test_labels_list)
    print(f'Accuracy: {accuracy}')

    # 保存预测结果
    predicted_labels = predicted.cpu().numpy()
    np.save(os.path.join(output_dir, 'predicted_labels.npy'), predicted_labels)

    # 保存混淆矩阵等其他可能的评估指标


if __name__ == '__main__':
    # 定义超参数
    input_size = 151296
    num_classes = 21
    num_epochs = 50000
    learning_rate = 0.00001

    # 训练模型
    fit_module(input_size, num_classes, num_epochs, learning_rate)

    model_path = "/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pth_files/05_20_152064_newlabel_CV_5.pth"
    test_data_path = r"/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_feature_all_data.pkl"
    labels_path = r"/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_label_all_data.pkl"
    output_dir = r"text_png_save/outcome_png"
    # 测试模型
    test_module(input_size, num_classes, model_path, test_data_path, labels_path, output_dir)