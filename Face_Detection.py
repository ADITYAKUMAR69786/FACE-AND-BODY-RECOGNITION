import cv2 as cv

capture = cv.VideoCapture(0)
# to open Camera

# accessing pretrained model
pretrained_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    boolean, frame = capture.read()
    if boolean == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2)

        # drawing rectangle in frame
        for (x, y, w, h) in coordinate_list:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display detected face
        cv.imshow("Live Face Detection", frame)

        # condition to break out of while loop
        if cv.waitKey(20) == ord('x'):
            break

capture.release()
cv.destroyAllWindows()

