import cv2
import numpy as np
import math
from picamera.array import PiRGBArray
from picamera import PiCamera
from SetupCar import *
import time

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 25
rawCapture = PiRGBArray(camera, size=(640, 480))

y = 150
factor = 1.2
minsp = 20

# allow the camera to warmup
time.sleep(0.1)


def nothing(x):
   pass

"""

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing"""
def process(frame):
   origin = frame.copy()
   frame = cv2.resize(frame, (320, 160))
   
   frame = frame[60:160, 0:320]
   
   ##### Tracking bar ######
   """
l_h = cv2.getTrackbarPos("L - H", "Trackbars")
   l_s = cv2.getTrackbarPos("L - S", "Trackbars")
   l_v = cv2.getTrackbarPos("L - V", "Trackbars")
   u_h = cv2.getTrackbarPos("U - H", "Trackbars")
   u_s = cv2.getTrackbarPos("U - S", "Trackbars")
   u_v = cv2.getTrackbarPos("U - V", "Trackbars")
"""
   
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   ### Blue object #####
   lower = np.uint8([0, 0, 150])
   upper = np.uint8([100, 255, 255])
   white_mask = cv2.inRange(hsv, lower, upper)
   
   result = cv2.bitwise_and(frame, frame, mask = white_mask)

   #cv2.imshow('inrange', result)

   ################ Detect lane #################

   gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

   ret, thresh = cv2.threshold(gray , 150,255, cv2.THRESH_BINARY_INV)
   cv2.imshow('thresh_hold', thresh)

   ima, contours, hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


   count = 0
   sum_cx = 0
   for contour in contours:
      area = cv2.contourArea(contour)
      print(area)
      if area < 8000:
         continue
      cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
      M = cv2.moments(contour)
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      sum_cx = sum_cx + cx
      count = count + 1
      # print('here')
   
   #print('count= ', count)
   if count > 0:
      X = sum_cx / count
      #print('X = ', X)
      cv2.circle(frame ,(int(X), 20), 10, (0,0,255), -1)
      angle = GetAngle(X)
      print('Angle: ', angle)
      speed = GetSpeed(angle)
   cv2.imshow('contour', frame)

# Calculate Angle
def GetAngle(x, xshape = 160):
   value = math.atan2((x-xshape), y)
   result = value * 180 / math.pi
   result = result * factor
   new_result = (result / 30) * 2
   turnServo(result)
   print('goc lech = ', result)
   return result;

# Calculate Speed
def GetSpeed(angle):
   speed = (abs(angle) / 45) * (100 - minsp)
   setSpeed(minsp, minsp)
   return minsp;
      
def main():
   print("Main")
   #motor_right.forward(minsp)
   #motor_left.forward(minsp)
   time.sleep(2)
   for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
   # while (True):
      # image = cv2.imread('252.png')
      image = frame.array
      # ret, image = cap.read()
      timeStart = time.time()
      process(image)
      print('1 frame: ', time.time() - timeStart)
      # clear the stream in preparation for the next frame
      rawCapture.truncate(0)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break
   
   # cap.release()
   cv2.destroyAllWindows()
   setSpeed(0,0)

if __name__ == "__main__":
   main()