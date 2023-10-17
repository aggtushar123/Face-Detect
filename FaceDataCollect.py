#Read a video from web cam using OpenCV
# Face Detection in Video 
# Click 20 pictures of the person who comes in the front of camera and savethem as numpy


import cv2
import numpy as np 

#Create a camera Object
cam = cv2.VideoCapture(0)

#Ask the name 
fileName = input("Enter the name of the person:")
dataset_path = "./data/"
offset = 20

#Model
model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

faceData = []
skip =0 

#Read image from camera object
while True:
    success, img = cam.read()
    if not success:
        print("Reading camera failed")
    
    #Store gray image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img,1.3,5)
    #pick the face with largest area

    faces = sorted(faces, key= lambda f:f[2]*f[3])
    #pick the largest 

    if len(faces)>0:
        f = faces[-1]
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)

    #crop and save the largest face
        cropped_face = img[y - offset : y+h + offset , x:x+w]

        cropped_face = cv2.resize(cropped_face, (100,100))
        skip+=1

        if skip % 10 == 0:
            faceData.append(cropped_face)
            print("Saved sofar" + str(len(faceData)))

    cv2.imshow("Image Window", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


#Write the faceData on the disk
faceData = np.asarray(faceData)
print(faceData.shape)
m = faceData.shape[0]
faceData = faceData.reshape((m,-1))

print(faceData.shape)

file = dataset_path + fileName + ".npy"
np.save(file, faceData)


cam.release()
cv2.destroyAllWindows()
