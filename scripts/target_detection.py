import cv2
import numpy as np
from kudrone_py_utils import imshow_r
def detect_target(image, red_range=10, min_radius=1, max_radius=400):
    image_dilate = cv2.dilate(image, np.ones((3,3), np.uint8))

    image_dilate_inv = cv2.bitwise_not(image_dilate)
    
    hsv_inv = cv2.cvtColor(image_dilate_inv, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([90-red_range, 70, 50])
    upper_bound = np.array([90+red_range, 255, 255])
    thresh_red = cv2.inRange(hsv_inv, lower_bound, upper_bound)

    thresh_red_blur = cv2.GaussianBlur(thresh_red, (7,7), 2)
    detected_circles = cv2.HoughCircles(thresh_red_blur, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 100,
               param2 = 30, minRadius = min_radius, maxRadius = max_radius)
    
    return detected_circles

    

if __name__ == "__main__":
    image = cv2.imread("/home/batu/Desktop/drone/2022-07-19 00:04:00.565641-out.jpg")

    detected_circles = detect_target(image)
    print(detected_circles)
    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        print(detected_circles)
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
    
            # Draw the circumference of the circle.
            cv2.circle(image, (a, b), r, (0, 255, 0), 2)
    
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
    imshow_r("Detected Circles", image, (1600, 900))
    cv2.waitKey(0)        
    # cv2.imshow("img", thresh_red_blur)
    while cv2.waitKey(1) != ord("q"):
        pass
    cv2.destroyAllWindows()