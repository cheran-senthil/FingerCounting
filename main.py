import numpy as np
import cv2
from sklearn import tree

X = np.load('X.npy').reshape(-1, 64*64)
Y = np.load('Y.npy')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (64, 64))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    clf.predict([gray.ravel()])

cap.release()
cv2.destroyAllWindows()
