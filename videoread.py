import cv2

#Create Camera Object
cam = cv2.VideoCapture(0)


#Read image from Camera object
success, img = cam.read()

if not success:
    print("Reading Camera failed")

cv2.imshow("Image Window", img)

cv2.waitKey(4000)