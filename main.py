from SetupCar import *
from detect_line import *
from picamera import PiCamera
from picamera.array import PiRGBArray

UH = 35
US = 120
UV = 255
LH = 20
LS = 50
LV = 200

camera = PiCamera()
camera.framerate = 20
camera.resolution = (688, 400)
rawCapture = PiRGBArray(camera)
sleep(2)


creatHSV()
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    image = frame.array
    UH = cv2.getTrackbarPos('UH','UpperHSV')
    US = cv2.getTrackbarPos('US','UpperHSV')
    UV = cv2.getTrackbarPos('UV','UpperHSV')
    LH = cv2.getTrackbarPos('LH','LowerHSV')
    LS = cv2.getTrackbarPos('LS','LowerHSV')
    LV = cv2.getTrackbarPos('LV','LowerHSV')
    binary_image =  binary_cvt(image, np.array([LH,LS,LV]),np.array([UH,US,UV]))
    cv2.imshow("binary_image", binary_image)
    rawCapture.truncate(0)
    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break


turnLED1(1)
print("exit adjust HSV range")
cv2.destroyAllWindows()

for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    image = frame.array
    #processing image
    blur = cv2.bilateralFilter(image,9,75,75)
    #cv2.imshow("blur", blur)
    
    # convert to binary
    binary_image =  binary_cvt(blur, np.array([LH,LS,LV]),np.array([UH,US,UV]))
    #bird_view = warp_image(binary_image)
    
    cv2.imshow("image for processing", binary_image)
    
    left_fit,right_fit = track_lanes_initialize(binary_image)
    left_fit,right_fit = check_missing_line(left_fit,right_fit)
    center_point = get_center(left_fit,right_fit)
    angle = int(errorAngle(center_point))
    print(angle)
    turnServo(angle)
    setSpeed(20,20)
    rawCapture.truncate(0)
    # if "ESC" is pressed then exit
    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break

turnLED1(0)
turnServo(0)
setSpeed(0,0)
cv2.destroyAllWindows()

