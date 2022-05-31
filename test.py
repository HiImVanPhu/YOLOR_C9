import cv2
import time

# cap = cv2.VideoCapture(r"rtsp://admin:bk123456@192.168.9.135:554/Streaming/Channels/1/")
cap = cv2.VideoCapture(r"E:\CameraSanslab\28_04_22\2022-05-19\ch01_00000000061000100.mp4")


# cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
count = 0
while True:
    count += 1
    ret, frame = cap.read()
    # frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow("image", frame)
    key = cv2.waitKey(100)
    if key == ord("q"):
        break
    # if count == 25:
    #     time.sleep(4)
    #     count += 4 * 25
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, count)

# import tensorflow as tf
#
# print(tf.__version__)
