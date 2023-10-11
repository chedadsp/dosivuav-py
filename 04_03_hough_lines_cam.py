import cv2 as cv
import numpy as np

camera = cv.VideoCapture(2)
while True:
    ret, img = camera.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 170, apertureSize=3)

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=200, maxLineGap=10)

    if lines is not None:
        for points in lines:
            x1, y1, x2, y2 = np.array(points[0])
            cv.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv.imshow("Hough lines", edges)
    if cv.waitKey(30) & 0xFF == 27: break

camera.release()
cv.destroyAllWindows()