from Lane import *
import cv2
image = cv2.imread("fx_UIT_Car1.png")
lane = Lane(image)
lane.create_Trackbar()

if __name__ == "__main__":
    while(True):
        lane.get_Lines()
        print(lane.left_line,lane.right_line)
        lane.get_center_line_unwarped()
        angle = int(errorAngle(lane.center_line))
        speed = int(calcul_speed(angle))
        # print(lane.left_line,lane.right_line)
        lane.draw_lane()
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    pass