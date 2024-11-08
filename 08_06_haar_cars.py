import cv2 as cv

cap = cv.VideoCapture('vid/cars.mp4')
#cap = cv.VideoCapture('vid/slow_traffic_small.mp4')
#cap = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar/cars.xml')

while cap.isOpened():
    ret, img = cap.read()
    if not ret: break

    #img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x+w, y+h), 255, 1)

    cv.imshow('Cars', img)
    
    if cv.waitKey(30) & 0xFF == 27: break

cap.release()
cv.destroyAllWindows()
