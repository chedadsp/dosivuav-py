import numpy as np
import cv2 as cv

def getTemplateMatch(img, gray, templateFileName):
    template = cv.imread(templateFileName, 0)
    matches = cv.matchTemplate(gray, template, method=cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matches)
    cv.rectangle(img, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 0, 255), 2)
    cv.imshow('Find a car?',img)
    cv.waitKey(0)


img = cv.imread("img/red_car.jpg")
cv.imshow('Find a car?',img)
cv.waitKey(0)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

getTemplateMatch(img, gray, "img/template01.jpg")
getTemplateMatch(img, gray, "img/template02.jpg")
getTemplateMatch(img, gray, "img/template03.jpg")

cv.destroyAllWindows()