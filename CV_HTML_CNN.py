import os
import pickle
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
def fit_module(input_size, num_classes, num_epochs, learning_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir_path = '/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/0520_labels'
    # features_list, labels_list, filepath_list = extract_features(dir_path)

    # 给出路径去提取特征，将提取的特征存在一个pkl文件，避免二次提取
    # with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_feature_0520_CV.pkl', 'wb') as file:
    #     pickle.dump(features_list, file)
    # with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_label_0520_CV.pkl', 'wb') as file:
    #     pickle.dump(labels_list, file)
    # with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_filepath_0520_CV.pkl', 'wb') as file:
    #     pickle.dump(filepath_list, file)

    # 使用特征文件
    with open(
            '/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_feature_0520_CV.pkl',
            'rb') as file:
        features_list = pickle.load(file)
    with open(
            '/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_label_0520_CV.pkl',
            'rb') as file:
        labels_list = pickle.load(file)

    # 创建模型实例并移动到GPU
    model = Classifier(input_size, num_classes).to(device)
    # model.load_state_dict(torch.load("/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pth_files/05_20_152064_newlabel_CV_4.pth"))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    feature_array = np.array(features_list)
    feature = torch.tensor(feature_array).to(device)
    label = labels_list
    #
    # 训练模型
    for epoch in range(num_epochs):
        # 创建一个从字符串到整数的映射（标签编码）
        label_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                         '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19,
                         '20': 20}
        # label_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7,'9': 8, '10': 9}
        # 使用映射将字符串标签转换为整数标签
        labels = [label_mapping[label_str] for label_str in label]

        # 现在将整数标签列表转换为PyTorch张量
        labels_tensor = torch.tensor(labels, dtype=torch.long)

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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    torch.save(model.state_dict(),
               "/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pth_files/05_20_152064_newlabel_CV_5.pth")


def test_module(input_size, num_classes, num_epochs, learning_rat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Classifier(input_size, num_classes).to(device)
    model.load_state_dict(torch.load(
        "/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pth_files/05_20_152064_newlabel_CV_5.pth"))
    model.eval()

    # dir_path = '/root/fraud_webpage_classification/public_data/all_data'
    # test_features_list, test_labels_list, filepath_list = white_extract_features(dir_path)
    # #
    # with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_feature_all_data.pkl', 'wb') as file:
    #     pickle.dump(test_features_list, file)
    # with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_label_all_data.pkl', 'wb') as file:
    #     pickle.dump(test_labels_list, file)
    # with open('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_filepath_all_data.pkl', 'wb') as file:
    #     pickle.dump(filepath_list, file)
    # # #
    # # 加载文件中的数据

    with open(
            rf'/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_feature_all_data.pkl',
            'rb') as file:
        test_features_list = pickle.load(file)
    with open(
            rf'/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_label_all_data.pkl',
            'rb') as file:
        test_labels_list = pickle.load(file)
    with open(
            rf'/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/__ulis__/__init__/__lib__/pkl_files/temp_filepath_all_data.pkl',
            'rb') as file:
        filepath_list = pickle.load(file)

    # # 从原始列表中随机选择800个数据
    # random_data =[(test_features_list[i], test_labels_list[i], filepath_list[i]) for i in random.sample(range(len(test_features_list)), 5000)]
    # test_features_list, test_labels_list, filepath_list= [_[0] for _ in random_data], [_[1] for _ in random_data], [_[2] for _ in random_data]

    test_feature_array = np.array(test_features_list)
    test_feature = torch.tensor(test_feature_array).to(device)
    test_label = test_labels_list

    with torch.no_grad():
        label_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                         '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17,
                         '18': 18, '19': 19, '20': 20, '100': 100}
        # 使用映射将字符串标签转换为整数标签
        labels_vec = [label_mapping[label_str] for label_str in test_label]
        # 现在将整数标签列表转换为PyTorch张量
        labels_tensor_vec = torch.tensor(labels_vec, dtype=torch.long)
        # 如果GPU可用，将标签张量移动到GPU上
        if torch.cuda.is_available():
            labels_tensor_vec = labels_tensor_vec.to(device)

        # 前向传播
        outputs_vec = model(test_feature)
        max_value, predicted = torch.max(outputs_vec, dim=1)

        y_pred = []
        t_pred = []
        y_file_path = []
        max_value_list = []
        for i in range(len(max_value)):
            print(max_value[i].item(), ': ', test_labels_list[i])
            if int(max_value[i].item()) >= 0:  # 距离阈值，距离越高，模型预测的相似性越高。但是太高也有过拟合的情况，但是可以忽略
                y_pred.append(predicted[i].item())
                t_pred.append(int(test_labels_list[i]))
                y_file_path.append(filepath_list[i])
                max_value_list.append(max_value[i].item())

        print(len(y_file_path))
        for i in range(len(y_file_path)):
            # file_name = test_labels_list[i]+"_"+str(max_value_list[i])[:4]+"_"+y_file_path[i].split('/')[-1]
            file_name = y_file_path[i].split('/')[-1]
            if not os.path.isdir(rf'text_png_save/outcome_png/{predicted[i]}'):
                os.makedirs(rf'text_png_save/outcome_png/{predicted[i]}')
            shutil.copy(filepath_list[i], rf'text_png_save/outcome_png/{predicted[i]}/{file_name}')


if __name__ == '__main__':
    # 定义超参数
    input_size = 151296
    num_classes = 21
    num_epochs = 50000
    learning_rate = 0.00001

    # 训练模型
    fit_module(input_size, num_classes, num_epochs, learning_rate)

    # 测试模型
    test_module(input_size, num_classes, num_epochs, learning_rate)
