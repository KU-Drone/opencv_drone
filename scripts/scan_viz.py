import webcam_viz
import pickle
import cv2 

with open("../scan.pickle", "rb") as file:
    xy_bgr = pickle.load(file)

img = webcam_viz.sampleImg(xy_bgr, step=0.1)

cv2.imshow("img", img)
while True:
    if cv2.waitKey(10) == ord("q"):
        break