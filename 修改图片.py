import numpy as np
import cv2
import os
from PIL import Image
import skimage

"change pixel"
root = "D:/DataSet/3Dircadb/png/problemgt/1.20"
new_root = "D:/DataSet/3Dircadb/png/problemgt/new"
files = [i for i in os.listdir(root)]
for file in files:
    file_path = root + '/' + file
    input_file = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    #修改像素值
    input_file[input_file == 1] = 255
    #整体亮度修改
    # input_arr = skimage.exposure.adjust_gamma(input_arr, 0.5)

    new_file = new_root + '/' + file
    cv2.imwrite(new_file, input_file)

# "挑选"
# root = "D:/DataSet/LiTs/2.0_labelpng/"
# new_root = "D:/DataSet/LiTS/cancer/"
# files = [i for i in os.listdir(root)]
# for file in files:
#     file_path = root + file
#     # CTpath = 'D:/DataSet/LiTs/ct_15-20/' + file
#     # preCTpath = 'D:/DataSet/LiTs/Two_branch_1-10/previousCT/' + file
#     # new_CTpath = 'D:/DataSet/LiTs/liverct15-20/' + file
#     # new_preCTpath = 'D:/DataSet/LiTs/Two_branch_1-10/preliverCT/' + file
#     input_file = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
#     if 127 in input_file:
#         new_file = new_root + file
#         cv2.imwrite(new_file, input_file)
#         # ct_file = cv2.imread(CTpath, cv2.IMREAD_UNCHANGED)
#         # cv2.imwrite(new_CTpath, ct_file)
#         # prect_file = cv2.imread(preCTpath, cv2.IMREAD_UNCHANGED)
#         # cv2.imwrite(new_preCTpath, prect_file)
#         continue

# "旋转"
# root = "D:/DataSet/LiTS/test/11,12GT/"
# new_root = "D:/DataSet/LiTS/test/rotateGT/"
# files = [i for i in os.listdir(root)]
# for file in files:
#     file_path = root + '/' + file
#     img = Image.open(file_path)
#     img = img.transpose(Image.ROTATE_90)
#     new_file = new_root + '/' + file
#     img.save(new_file)