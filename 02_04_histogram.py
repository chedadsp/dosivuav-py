import cv2 as cv
from cv2 import COLOR_BGR2GRAY
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('img/car_and_road.jpg')
cv.imshow('Car', img)

gray = cv.cvtColor(img, COLOR_BGR2GRAY)

gray_hist = cv.calcHist([img], [0], None, [256], [0, 256])

plt.figure()
plt.title('Histogram of intensity')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])

plt.figure()
plt.title('Histogram per color')
#plt.xlabel('Bins')
#plt.ylabel('# of pixels')
plt.xlim([0, 256])

for i, col, in enumerate(('b', 'g', 'r')):
    histogramColor = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histogramColor, color= col)

plt.show()
