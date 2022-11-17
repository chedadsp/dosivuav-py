import cv2 as cv

camera = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

while True:
    ret, img = camera.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x+w, y+h), 255, 1)

    cv.imshow('Face', img)
    if cv.waitKey(30) & 0xFF == 27: break

camera.release()
cv.destroyAllWindows()
