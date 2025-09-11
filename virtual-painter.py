import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


################
brushThickness=15
eraserThickness= 50
################


folderPath='header-images'
mylist=os.listdir(folderPath)

overlay=[]
for impath in mylist:
    im=cv2.imread(os.path.join(folderPath, impath))
    overlay.append(im)

header=overlay[0]
drawColor=(255,0,0)

cap=cv2.VideoCapture(0)
cap.set(3,950)
cap.set(4,720)
detector=htm.HandDetector(detectionCon=0.85)


imgcanvas=np.zeros((540,960,3),np.uint8)
imgcanvas = cv2.flip(imgcanvas,1)

while True:
    #import image
    success,img=cap.read()
    img=cv2.flip(img,1)
    #print(img.shape)


    #Find hand landmarks
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        #print(lmList)

        x1,y1=lmList[8][1:] # tip of index finger
        x2,y2=lmList[12][1:]  # tip of middle finger



        #check which fingers are up
        fingers=detector.fingersUp()
        #print(fingers)

        #if selection mode-Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img,(x1,y1-18),(x2,y2+18),drawColor,cv2.FILLED)
            #print('selection mode')
            #checking for the click
            if y1<header.shape[0]+50:
                if 100<x1<260:
                    header=overlay[2]
                    drawColor=(255,0,255)
                if 320<x1<470:
                    header=overlay[0]
                    drawColor = (255, 0, 0)
                if 520<x1<700:
                    header=overlay[1]
                    drawColor = (0, 255, 0)
                if 720<x1<900:
                    header=overlay[3]
                    drawColor = (0, 0, 0)

        # if Drawing mode - Index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            #print('drawing mode')
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                xp,yp=0,0
                xp,yp=x1,y1
                cv2.line(img, (xp, yp), (x1, y1), drawColor,eraserThickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1

    imGray=cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)

    _,imgInv=cv2.threshold(imGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgcanvas)

















    #setting the header image
    h,w,_=header.shape
    img[0:h,0:w]=header


    cv2.imshow('Video',img)

    cv2.waitKey(1)


