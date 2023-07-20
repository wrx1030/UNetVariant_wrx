import cv2

# 读取图片并转至灰度模式
imagepath = 'D:/DataSet/3Dircadb1/test/3Dircadb1.8_image_91.png'
img = cv2.imread(imagepath, 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
# 图片轮廓
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
# 寻找凸包并绘制凸包（轮廓）
hull = cv2.convexHull(cnt)
print(hull)

length = len(hull)
for i in range(len(hull)):
    cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 2)

# 显示图片
cv2.imshow('line', img)
cv2.waitKey()
