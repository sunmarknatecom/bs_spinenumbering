import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy


paths = os.walk(".\\data5\\")

filePaths = []
for dir_, _, files_ in paths:
    if "labelData" in dir_:
        tempPath = os.listdir(dir_)[0]
        filePaths.append(dir_+"\\"+tempPath)

areas = []

for paths in filePaths2:
    tempObj = np.load(paths)
    contours, h_ = cv2.findContours(tempObj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp_arr = []
    for elem in contours:
        temp_va = cv2.contourArea(elem)
        temp_arr.append(temp_va)
    temp_np = np.array(temp_arr)
    idx = np.argmax(temp_np)
    x, y, w, h = cv2.boundingRect(contours[idx])
    print(cv2.contourArea(contours[idx]), idx, x, y, w, h)
    areas.append(cv2.contourArea(contours[idx]))
