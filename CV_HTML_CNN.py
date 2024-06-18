
import pickle

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
    features_list, labels_list, filepath_list = extract_features(dir_path, '/root/fraud_webpage_classification/public_data/BlackWebHTML')
    with open('temp_feature_train.pkl', 'wb') as file:
        pickle.dump(features_list, file)
    with open('temp_label_train.pkl', 'wb') as file:
        pickle.dump(labels_list, file)
    with open('temp_filepath_train.pkl', 'wb') as file:
        pickle.dump(filepath_list, file)

    with open('temp_feature_train.pkl', 'rb') as file:
        features_list = pickle.load(file)
    with open('temp_label_train.pkl', 'rb') as file:
        labels_list = pickle.load(file)

    # 创建模型实例并移动到GPU
    model = Classifier(input_size, num_classes).to(device)
    # model.load_state_dict(torch.load("05_11_151296_10class_model_2.pth"))
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    feature_array = np.array(features_list)
    feature = torch.tensor(feature_array).to(device)
    label = labels_list

    # 训练模型
    for epoch in range(num_epochs):
        # 创建一个从字符串到整数的映射（标签编码）
        # label_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,'9': 9, '10': 10}
        label_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7,'9': 8, '10': 9}
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    torch.save(model.state_dict(), "05_11_151296_10class_model_2.pth")


def test_module(input_size, num_classes, num_epochs, learning_rat):
    model = Classifier(input_size, num_classes).to(device)
    model.load_state_dict(torch.load("05_11_151296_10class_model_END.pth"))
    model.eval()
    total_correct = 0
    total_samples = 0

    # 如果没有处理好的特征，则开启这段代码提取特征
    # dir_path = '/root/fraud_webpage_classification/public_data/BlackWebPNGS_3'
    # test_features_list, test_labels_list, filepath_list = extract_features(dir_path, '/root/fraud_webpage_classification/public_data/NO_repeat_pngs')
    # #
    # with open('temp_feature_PNGS_3.pkl', 'wb') as file:
    #     pickle.dump(test_features_list, file)
    # with open('temp_label_PNGS_3.pkl', 'wb') as file:
    #     pickle.dump(test_labels_list, file)
    # with open('temp_filepath_PNGS_3.pkl', 'wb') as file:
    #     pickle.dump(filepath_list, file)

    # 加载文件中的数据
    with open('temp_feature_PNGS_3.pkl', 'rb') as file:
        test_features_list = pickle.load(file)
    with open('temp_label_PNGS_3.pkl', 'rb') as file:
        test_labels_list = pickle.load(file)
    with open('temp_filepath_PNGS_3.pkl', 'rb') as file:
        filepath_list = pickle.load(file)

    # 转换特征格式
    test_feature_array = np.array(test_features_list)
    test_feature = torch.tensor(test_feature_array).to(device)
    test_label = test_labels_list

    with torch.no_grad():
        label_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9}
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
        for i in range(len(max_value)):
            if max_value[i].item() > 3:
                y_pred.append(predicted[i].item())
                t_pred.append(int(test_labels_list[i]) - 1)
                if predicted[i].item() + 1 == int(test_labels_list[i]):
                    print(max_value[i].item(), end="  :")
                    print(predicted[i].item() + 1, end="  :")
                    print(int(test_labels_list[i]))
        # 统计预测结果
        total_samples += labels_tensor_vec.size(0)
        total_correct += (predicted == labels_tensor_vec).sum().item()

    accuracy = calculate_accuracy(t_pred, y_pred)
    print("准确率：", accuracy)
    print("命中率：", len(y_pred) / len(predicted))

    accuracy = total_correct / total_samples
    print(rf"所有样本准确率: {accuracy}")




# 定义超参数
input_size = 151296
num_classes = 10
num_epochs = 50000
learning_rate = 0.00001
# # 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_path = '/root/fraud_webpage_classification/public_data/NO_repeat_old'

# 训练模型
fit_module(input_size, num_classes, num_epochs, learning_rate)

# 测试模型
test_module(input_size, num_classes, num_epochs, learning_rate)

