import cv2

cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


while True:
    success, img = cam.read()

    faces = model.detectMultiScale(img,1.3,5)

    for f in faces:
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)

    cv2.imshow("Image Window", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destryAllWindows()
