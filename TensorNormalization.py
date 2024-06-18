import traceback
from paddleocr import PaddleOCR
import re
from transformers import AutoTokenizer, AutoModel
from zhon.hanzi import punctuation
import jieba
import joblib
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
import torch
from skimage.exposure import exposure
from skimage.feature import hog
from transformers import ViTImageProcessor, ViTModel
import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CV_processor = ViTImageProcessor.from_pretrained('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/google/vit-base-patch16-224-in21k')
CV_model = ViTModel.from_pretrained('/root/fraud_webpage_classification/project_gjr/fraud_webpage_classification/google/vit-base-patch16-224-in21k').to(device)

def math_min_sum(gray_image):
    pixels = gray_image.flatten()
    min_value = np.min(pixels)
    max_value = np.max(pixels)
    mean_value = np.mean(pixels)
    min_sum = min([mean_value - min_value, max_value - mean_value])
    if min_sum <= 3:
        return False
    return True


def resize_picture(picture_path):
    img = cv2.imread(picture_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(picture_path)  # 读取彩色图像
    if img.shape[1] > 1280 or img.shape[0] > 720:
        img = img[:720, :1280]
        img_color = img_color[:720, :1280]
    roi_pil = Image.fromarray(np.uint8(img_color)).convert('RGB')
    return roi_pil


def extract_features(dir_path, html_dir_path):
    features_list = []
    label_list = []
    filepath_list = []

    for _ in sorted(os.listdir(dir_path)):
        nb_path = os.path.join(dir_path, _)
        if os.path.basename(nb_path) == '.uuid':
                continue
        for __ in tqdm(os.listdir(nb_path)):
            _png_path = os.path.join(nb_path, __)
            if os.path.basename(_png_path) == '.uuid':
                continue
            if not math_min_sum(_png_path):
                continue
            try:
                '''CV特征 '''
                CV_image = resize_picture(_png_path)
                CV_inputs = CV_processor(images=CV_image, return_tensors="pt").to(device)
                CV_outputs = CV_model(**CV_inputs)

            except Exception as e:
                traceback.print_exc()
                continue

            flattened_tensor = CV_outputs[0].cpu().detach().view(1, -1)
            flattened_list = flattened_tensor[0]
            features_list.append(flattened_list)
            label_list.append(_)
            filepath_list.append(_png_path)
    return features_list, label_list, filepath_list
