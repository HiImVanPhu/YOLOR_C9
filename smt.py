import cv2

cap = cv2.VideoCapture(r"D:\30_31-1-21-c9\1.mp4")
cap.set(cv2.CAP_PROP_FPS, 10)

while True:
    ret, img = cap.read()
    cv2.imshow("image", img)
    key = cv2.waitKey(100)
    if key == ord("q"):
        break
