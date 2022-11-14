import numpy as np
import cv2 as cv

image = cv.imread("img/car_and_road.jpg")

cv.imshow("Original image", image)

gaussian = cv.GaussianBlur(image, (3, 3), 0)
cv.imshow("Gussian blur", gaussian)

# Make high pass kernel
kernel_3x3 = np.ones((3, 3), np.float32) * -1
kernel_3x3[1, 1] = 9
print(kernel_3x3)

sharpen = cv.filter2D(gaussian, -1, kernel_3x3)
cv.imshow("3x3 Sharpen", sharpen)
cv.waitKey(0)

cv.destroyAllWindows()