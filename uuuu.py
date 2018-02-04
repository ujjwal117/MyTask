import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(4, 480)
cap.set(3, 240)
cap.set(15, 0.1)
while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    lower_blue = np.array([100,100,100])
    upper_blue = np.array([165,255,180])

    lower_orange = np.array([0, 140, 200])
    upper_orange = np.array([19, 255, 255])
    
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    kernel = np.ones((15,15),np.float32)/225
    
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    blur = cv2.GaussianBlur(erosion,(15,15),0)
    cv2.imshow('Gaussian Blurring',blur)

    cv2.imshow('frame',frame)
    #cv2.imshow('erosion',erosion)
    #cv2.imshow('dilation',dilation)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
