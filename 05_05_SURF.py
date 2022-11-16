import numpy as np
import cv2 as cv

img = cv.imread("img/car_and_road.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

surf = cv.xfeatures2d.SURF_create(5000)

features, descriptors = surf.detectAndCompute(gray, None)
img = cv.drawKeypoints(img, outImage=img, keypoints=features, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("SURF", img)

cv.waitKey(0)
cv.destroyAllWindows()