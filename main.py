import cv2
import numpy as np
from sklearn import tree

X = np.load('X.npy').reshape(-1, 64 * 64)
Y = np.load('Y.npy')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

cap = cv2.VideoCapture(0)
old_prediction = np.zeros((10))

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (64, 64))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    new_prediction = clf.predict([gray.ravel()])
    if np.any(new_prediction != old_prediction):
        print('Count of Fingers: %d' % np.argmax(new_prediction))
        old_prediction = new_prediction

cap.release()
cv2.destroyAllWindows()
