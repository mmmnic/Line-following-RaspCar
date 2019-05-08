import cv2
import numpy as np
import math

UH = 0
US = 0
UV = 0
LH = 0
LS = 0
LV = 0

xRatio = 0.25
def nothing(x):
    pass

def creatHSV():
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
    cv2.setTrackbarPos('UH', 'UpperHSV', 60)
    cv2.setTrackbarPos('US', 'UpperHSV', 160)
    cv2.setTrackbarPos('UV', 'UpperHSV', 255)
    cv2.setTrackbarPos('LH', 'LowerHSV', 0)
    cv2.setTrackbarPos('LS', 'LowerHSV', 0)
    cv2.setTrackbarPos('LV', 'LowerHSV', 200) 
    return;
    
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
    [xRatio * x, y],
    [xRatio * x, 0],
    [x - (xRatio * x), 0],
    [x - (xRatio * x), y]
    ])
    
    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)
    warped_img = cv2.warpPerspective(img, inverse_perspective_transform, image_size, flags=cv2.INTER_LINEAR)

    return warped_img
def track_lanes_initialize(binary_warped):   
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint+100])
    rightx_base = np.argmax(histogram[midpoint+100:]) + midpoint+100
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 60
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []  
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    if len(left_lane_inds) != 0:
        left_lane_inds = np.concatenate(left_lane_inds)
    if len(right_lane_inds) !=0:
        right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.array([])
    right_fit = np.array([])
    if len(leftx) > 100:
        left_fit  = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 100:
        right_fit  = np.polyfit(righty, rightx, 2)
    return left_fit,right_fit
def check_missing_line(left_fit,right_fit):
    if len(left_fit) == 0:
        left_fit = np.array([0,0,0])
    if len(right_fit) == 0:
        right_fit = np.array([0,0,640])
    return left_fit,right_fit
def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]
def get_center(left_fit, right_fit, shape = 480):
    # ploty = np.linspace(0, shape - 1, shape)
    leftx =  get_val(0,left_fit)
    rightx =  get_val(0,right_fit)
    centerx = int((leftx + rightx)/2)
    return (centerx,0)
def errorAngle(center_point):
    carPosx , carPosy = 320, 480
    dstx, dsty = center_point[0],center_point[1]
    # print(carPosx,carPosy)
    if dstx == carPosx:
        return 0
    if dsty == carPosy:
        if dstx < carPosx:
            return -45
        else:
            return 45
    pi = math.acos(-1.0)
    dx = dstx - carPosx
    dy = carPosy - dsty
    if dx < 0: 
        angle = (math.atan(-dx / dy) * -180 / pi)/2.5
        if angle >= 28 or angle <= -28: # maybe must turn 90
            if angle > 0:
                return 45
            return -45
        return angle
    #################################################
    angle = (math.atan(dx / dy) * 180 / pi)/2.5
    if angle >= 25 or angle <= -25: # maybe must turn 90
        if angle > 0:
            return 45
        return -45
    return angle

