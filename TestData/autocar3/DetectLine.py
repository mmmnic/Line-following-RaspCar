import cv2
import numpy as np
import math

xRatio = 0.2
yRatio = 0.5
UH = 255
US = 150
UV = 255
LH = 0
LS = 0
LV = 0


def binary_cvt(image, lower, upper):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask
	
def warp_image(img):
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]
    #the "order" of points in the polygon you are defining does not matter
    #but they need to match the corresponding points in destination_points!
    ## my source
    source_points = np.float32([
    [0, y],
    [0, 0],
    [x, 0],
    [x, y]
    ])
    
    destination_points = np.float32([
    [xRatio*x, y*yRatio],
    [xRatio*x, 0],
    [x - (xRatio*x), 0],
    [x - (xRatio*x), y*yRatio]
    ])
    
    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    #inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)
    warped_img = cv2.warpPerspective(img, perspective_transform, image_size)
    return warped_img

img = cv2.imread('roadLines.jpg', 1)
#resizeImg = cv2.resize(img, (640, 480))
#hsv = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2HSV)
bird_view = warp_image(img)

cv2.imshow('bird_view', bird_view)
cv2.waitKey(1)
