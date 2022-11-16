import numpy as np
import cv2 as cv

img = cv.imread("img/car_and_road.jpg")

gray = np.float32(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

corners = cv.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=50)

for corner in corners:
    x, y = corner[0]
    x = int(x - 10)
    y = int(y - 10)
    cv.rectangle(img, (x, y), (x + 20, y + 20), (255, 127, 127,), 2)

cv.imshow("Corners", img)

cv.waitKey(0)
cv.destroyAllWindows()