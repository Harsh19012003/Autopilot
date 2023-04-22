import cv2
from operator import itemgetter
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpllimg
import numpy as np
frame = cv2.imread("frame.jpeg")
'''cv2.resize(frame, (416, 416))
frame = frame[50: 365, :]'''
frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
print(frame.shape)
cv2.imshow("a", frame)

# Coordinates that you want to Perspective Transform
# top left, top right, bottom left, bottom right
# pts1 = np.float32([[112.4, 125],  [298.1, 135], [-230.9, 416], [637.8, 416]]) of solecthon  top left, top right, bottom left, bottom right
pts1 = np.float32([[103, 207], [258,206], [-779.2,416], [1597.4, 416]]) 
# best with blackstrip
# pts1 = np.float32([[112.43462270113994, 144.35328009613204], [298.0904091992343, 151.94741233299698], [-328.4842423217328, 416.0], [724.9171873465214, 416.0]])
pts2 = np.float32([[0, 0], [416, 0], [0, 416], [416, 416]])

external_points_set = [[0, 0], [200, 200], [100, 100], [135, 265], [150, 200], [299, 150], [389, 473], [378, 465], [500, 500]]
# fv_points = np.array(external_points_set,np.uint8)
fv_points = np.float32(external_points_set)

print("fv_points: ",fv_points)
# C:\Users\Gautham\Downloads\Autopilot\Autopilot\perspective_transform.py
img = np.zeros((416,416),dtype=np.uint8)
for fv in fv_points:
    cv2.circle(img,(int(fv[0]),int(fv[1])),5,(255,0,0),-1)
cv2.imshow("fv",img)
cv2.waitKey(0)


matrix = cv2.getPerspectiveTransform(pts1, pts2)
'''matrix_manual = [[-2.12396813e-01, -8.96538138e-01,  2.07460266e+02],
 [-1.48949900e-02, -2.30872344e+00,  4.79439937e+02],
 [-3.58052628e-05, -5.18322980e-03,  1.00000000e+00]]'''
print("matrix :", type(matrix))

tv = []
for i in fv_points:
    a = np.array([[i]])
    # a = np.array([a])
    pointsOut = cv2.perspectiveTransform(a,matrix)
    box = int(pointsOut[0][0][0]),int(pointsOut[0][0][1])
    print(box)
    tv.append(box)
print("box: ",tv)
for i in tv:
    cv2.circle(img,(i[0],i[1]),5,(255,0,0),-1)
# 
# hehe = cv2.getPerspectiveTransform(external_points_set, matrix)
# print("hehe: ", hehe)
# result = cv2.warpPerspective(fv_points, matrix, (416, 416))
# print("result: ",result)
# result_manual = cv2.warpPerspective(frame, matrix_manual, (416, 416))
cv2.imshow('c', img)
# cv2.imshow('c', result_manual)
cv2.waitKey(0)
# cv2.imwrite("result.jpg", result)
