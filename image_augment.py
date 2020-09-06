import cv2
import numpy as np


cap = cv2.VideoCapture('rtsp://admin:zhihe000@192.168.1.182:554/h264/ch1/main/av_stream')
# Define the codec and create VideoWriter object
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        (b, g, r) = cv2.split(frame)
        bH = (cv2.equalizeHist(b))
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))


        cv2.imshow("frame", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # Release everything if job is finished
cap.release()

cv2.destroyAllWindows()