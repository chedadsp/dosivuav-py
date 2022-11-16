import numpy as np
import cv2 as cv

img = cv.imread("img/car_and_road.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create()

features = orb.detect(gray, None)
features, descriptors = orb.compute(gray, features)
img = cv.drawKeypoints(img, outImage=img, keypoints=features, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("ORG", img)

cv.waitKey(0)
cv.destroyAllWindows()