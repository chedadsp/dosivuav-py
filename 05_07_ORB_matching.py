import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

object = cv.imread("img/template01.jpg")
# object = cv.imread("img/template03.jpg") # works with different orenation!
img = cv.imread("img/red_car.jpg")

obj_gray = cv.cvtColor(object, cv.COLOR_BGR2GRAY)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create()

features1, des1 = orb.detectAndCompute(obj_gray, None)
features2, des2 = orb.detectAndCompute(img_gray, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 20 matches.
img3 = cv.drawMatches(obj_gray,features1,img_gray,features2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

cv.waitKey(0)
cv.destroyAllWindows()