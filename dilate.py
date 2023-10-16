import os
import cv2
import numpy as np

def process_images(source_dir, target_dir):
    for file in os.listdir(source_dir):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_dir, file)
            img = cv2.imread(image_path, 0)
            kernel = np.ones((9, 9), np.uint8)
            dilate = cv2.dilate(img, kernel, iterations=1)
            os.makedirs(target_dir, exist_ok=True)
            filename = os.path.basename(file)
            cv2.imwrite(os.path.join(target_dir, filename), dilate)

# 指定源文件夹和目标文件夹的路径
source_directory = 'D:\\DataSet\\3Dircadb\\train\\GT'
target_directory = 'D:\\DataSet\\3Dircadb\\train\\GT_dilation'

process_images(source_directory, target_directory)