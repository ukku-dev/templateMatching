# pip install pyautogui
# pip install opencv-python
import cv2
import numpy as np
import pyautogui



cv2.namedWindow("result");
cv2.moveWindow("result", 0, 500);

img_piece = cv2.imread('tm1.png', cv2.IMREAD_COLOR)
h,w = img_piece.shape[:2]


while 1:
    pic = pyautogui.screenshot(region=(0, 0, 700, 500))
    img_frame = np.array(pic)
    img_frame  = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)
    meth = 'cv2.TM_CCOEFF'
    method = eval(meth)


    res = cv2.matchTemplate(img_piece, img_frame, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img_frame, top_left, bottom_right, (0, 255, 0), 2)
    print(max_val, top_left)

    cv2.imshow('result', img_frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

