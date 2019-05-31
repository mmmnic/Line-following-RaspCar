from SetupCar import *
from detect_line import *
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

def nothing(x):
        pass  
def create_Trackbar():
        cv2.namedWindow("lower")
        cv2.namedWindow("upper")
        cv2.createTrackbar('lowH','lower',0,179,nothing)
        cv2.createTrackbar('highH','upper',179,179,nothing)

        cv2.createTrackbar('lowS','lower',0,255,nothing)
        cv2.createTrackbar('highS','upper',255,255,nothing)

        cv2.createTrackbar('lowV','lower',0,255,nothing)
        cv2.createTrackbar('highV','upper',255,255,nothing)


camera = PiCamera()
camera.framerate = 30
camera.resolution = (640, 480)
sleep(2)
rawCapture = PiRGBArray(camera, size=(640, 480))


create_Trackbar()
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    lane = Lane(frame.array)
    lane.get_Lines()    
    rawCapture.truncate(0)
    if (cv2.waitKey(1) & 0xff == 27):
        break


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    tps = time.time()
    lane = Lane(frame.array)
    lane.get_Lines()
    #print(lane.left_line,lane.right_line)
    lane.get_center_line_unwarped()
    angle = int(errorAngle(lane.center_line))
    setSpeed(20, 20)
    turnServo(angle*1)
    #print('angle: ', angle)
    #print(lane.left_line,lane.right_line)
    lane.draw_lane()
    rawCapture.truncate(0)
    if (cv2.waitKey(1) & 0xff == 27):
        break
    print("tps la: ",time.time()- tps)

turnLED1(0)
turnServo(0)
setSpeed(0,0)
cv2.destroyAllWindows()

