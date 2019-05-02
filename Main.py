from picamera import PiCamera
from picamera.array import PiRGBArray
from SetupCar import *
import cv2
import numpy as np
import time


def nothing(x):
    pass

# Create trackbar and init value
cv2.namedWindow('UpperHSV')
cv2.namedWindow('LowerHSV')
cv2.createTrackbar('UH', 'UpperHSV', 0, 179, nothing)
cv2.createTrackbar('US', 'UpperHSV', 0, 255, nothing)
cv2.createTrackbar('UV', 'UpperHSV', 0, 255, nothing)
cv2.createTrackbar('LH', 'LowerHSV', 0, 179, nothing)
cv2.createTrackbar('LS', 'LowerHSV', 0, 255, nothing)
cv2.createTrackbar('LV', 'LowerHSV', 0, 255, nothing)
# init value
cv2.setTrackbarPos('UH', 'UpperHSV', 179)
cv2.setTrackbarPos('US', 'UpperHSV', 23)
cv2.setTrackbarPos('UV', 'UpperHSV', 220)
cv2.setTrackbarPos('LH', 'LowerHSV', 0)
cv2.setTrackbarPos('LS', 'LowerHSV', 0)
cv2.setTrackbarPos('LV', 'LowerHSV', 110) 
UH = cv2.getTrackbarPos('UH','UpperHSV')
US = cv2.getTrackbarPos('US','UpperHSV')
UV = cv2.getTrackbarPos('UV','UpperHSV')
LH = cv2.getTrackbarPos('LH','LowerHSV')
LS = cv2.getTrackbarPos('LS','LowerHSV')
LV = cv2.getTrackbarPos('LV','LowerHSV')


# Setup camera
camera = PiCamera()
camera.framerate = 30
camera.resolution = (1280, 720)
rawCapture = PiRGBArray(camera)

time.sleep(2)
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    
    frame = frame.array
    RSImg = cv2.resize(frame, (320, 300))
    setSpeed(0, 0)
    turnServo(0)
    # get frame
    cv2.imshow('origin', RSImg)
    
    # convert image to hsvhasattr
    hsv = cv2.cvtColor(RSImg, cv2.COLOR_BGR2HSV)
    # adjust trackbar
    UH = cv2.getTrackbarPos('UH','UpperHSV')
    US = cv2.getTrackbarPos('US','UpperHSV')
    UV = cv2.getTrackbarPos('UV','UpperHSV')
    LH = cv2.getTrackbarPos('LH','LowerHSV')
    LS = cv2.getTrackbarPos('LS','LowerHSV')
    LV = cv2.getTrackbarPos('LV','LowerHSV')
    
    # define range of color in HSV
    lower_HSV = np.array([LH,LS,LV])
    upper_HSV = np.array([UH,US,UV])

    # convert to binary image
    mask = cv2.inRange(hsv, lower_HSV, upper_HSV)
    
    # show image
    cv2.imshow('hsv', mask)

    rawCapture.truncate(0)
    # if "ESC" is pressed then exit
    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break;
    
cv2.destroyAllWindows()


