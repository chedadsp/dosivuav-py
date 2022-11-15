import cv2 as cv
import numpy as np

img = cv.imread('img/road.jpg')
cv.imshow('Original', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 170, apertureSize=3)
cv.imshow("Edges", edges)

lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=200, maxLineGap=10)

for points in lines:
    x1, y1, x2, y2 = np.array(points[0])
    cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv.imshow("Hough lines", img)

cv.waitKey(0)
cv.destroyAllWindows()