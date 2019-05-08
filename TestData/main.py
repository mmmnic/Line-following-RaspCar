from Lane import *
import cv2
image = cv2.imread("fx_UIT_Car1.png")
lane = Lane(image)
#Ham nay dung de tao HSV
lane.create_Trackbar()
#or dung
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
create_Trackbar()
if __name__ == "__main__":
    while(True):
        lane.get_Lines() 
        print(lane.left_line,lane.right_line)
        lane.get_center_line_unwarped()
        angle = int(errorAngle(lane.center_line))
        speed = int(calcul_speed(angle))
        # print(lane.left_line,lane.right_line)
        lane.draw_lane()
        lane.draw_center
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    pass
