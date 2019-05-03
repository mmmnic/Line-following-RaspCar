from SetupCar import *
from detect_line import *
from picamera import PiCamera
from picamera.array import PiRGBArray



camera = PiCamera()
camera.framerate = 20
camera.resolution = (680, 400)
rawCapture = PiRGBArray(camera)
sleep(2)

for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    image = frame.array
    # convert image to hsvhasattr
    #processing image
    blur = cv2.bilateralFilter(image,9,75,75)
    cv2.imshow("blur", blur)
    binary_image =  binary_cvt(blur, np.array([0,0,150]),np.array([179,121,255]))
    bird_view = warp_image(binary_image)
    left_fit,right_fit = track_lanes_initialize(bird_view)
    left_fit,right_fit = check_missing_line(left_fit,right_fit)
    center_point = get_center(left_fit,right_fit)
    angle = int(errorAngle(center_point))
    print(angle)
    cv2.imshow('bird_view', bird_view)
    turnServo(angle*2)
    setSpeed(20,20)
    rawCapture.truncate(0)
    # if "ESC" is pressed then exit
    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break
turnServo(0)
setSpeed(0,0)
cv2.destroyAllWindows()


