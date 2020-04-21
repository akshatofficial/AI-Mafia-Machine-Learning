import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
nose_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')
glasses = cv2.imread('glasses.png', -1)
mustache = cv2.imread('mustache.png', -1)

caps = cv2.VideoCapture(0)
while True:
    ret, frame = caps.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    face = face_cascade.detectMultiScale(frame, 1.3, 5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    for (x, y, w, h) in face:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for(ex, ey, ew, eh) in eyes:
            glasses = cv2.resize(glasses.copy(), (ew, eh))
            glass1 = cv2.resize(glasses.copy(), (int(1.1*ew), int(2.5*eh)))
            gw, gh, gc = glass1.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    if glass1[i, j][3] != 0:
                        roi_color[ey-int(eh/1.5)+i, int(ex)+j] = glass1[i, j]

        nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (nx, ny, nw, nh) in nose:
            mustache = cv2.resize(mustache.copy(), (nw, nh))
            mustache1 = cv2.resize(mustache.copy(), (nw, int(0.5*ny)))
            mw, mh, mc = mustache1.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    if mustache1[i, j][3] != 0:
                        roi_color[ny+int(nh/0.5)+i, nx+j] = mustache1[i, j]

    cv2.imshow('Frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

caps.release()
cv2.destroyAllWindows()
