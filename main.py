import numpy as np
import cv2
from sklearn import tree

X = np.zeros((1, 307200))
Y = [[0]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    clf.predict([gray.ravel()])

cap.release()
cv2.destroyAllWindows()
