import numpy as np
import cv2

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


# Insert image
img = cv2.imread('Line4.jpg', 1)
# scale to 15%
scale_percent = 15
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
small = cv2.resize(img, dim)


blur = cv2.bilateralFilter(small,9,75,75)
cv2.imshow('blur', blur)
# convert image to hsv
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


while True:
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


    lines = cv2.HoughLines(mask,1,np.pi/150,200)
    
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(small,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite('houghlines3.jpg',small)


    
    # show image
    cv2.imshow('hsv', small)

    # if "ESC" is pressed then exit
    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break;
    
cv2.destroyAllWindows()

