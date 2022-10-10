# pip install pyautogui
# pip install opencv-python
import cv2
import numpy as np
import pyautogui



cv2.namedWindow("result");
cv2.moveWindow("result", 0, 500);

img_obj = cv2.imread('box.png', cv2.IMREAD_COLOR)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_COLOR)
h,w = img_obj.shape[:2]


pic = pyautogui.screenshot(region=(0, 0, 700, 500))

sift = cv2.SIFT()

gray_scene= cv2.cvtColor(img_scene,cv2.COLOR_BGR2GRAY)
gray_obj= cv2.cvtColor(img_obj,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img_scene,None)
kp2, des2 = sift.detectAndCompute(img_obj,None)

#img3=cv2.drawKeypoints(img_obj,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img_scene,kp1,img_obj,kp2,good,None,flags=2)

cv2.imshow('result',img3)
cv2.waitKey(0)

    
