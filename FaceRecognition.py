import cv2
import numpy as np 
import os

dataset_path = "./data/"
faceData = []
labels =[]
nameMap = {}
offset = 20

classId = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):

        nameMap[classId] = f[:-4]
        #X-Values
        dataItem = np.load(dataset_path + f)
        m = dataItem.shape[0]
        faceData.append(dataItem)

        #Y-Values
        target = classId * np.ones((m,))
        classId+=1

        labels.append(target)


print(faceData)
print(labels)

XT = np.concatenate(faceData, axis=0)
yT = np.concatenate(labels, axis=0).reshape((-1,1))

print(XT.shape)
print(yT.shape)
print(nameMap)

#Algorithm
def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))

def knn(X,y,xt,k=5):

    m = X.shape[0]
    dlist =[]

    for i in range(m):
        d = dist(X[i],xt)
        dlist.append((d,y[i]))

    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:,1]

    labels, cnts = np.unique(labels, return_counts= True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

# Predictions

#Create a camera Object
cam = cv2.VideoCapture(0)


#Model
model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#Read image from camera object
while True:
    success, img = cam.read()
    if not success:
        print("Reading camera failed")
    

    faces = model.detectMultiScale(img,1.3,5)
    #pick the face with largest area

    #render a box around each face and predict its name
    for f in faces:
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)

    #crop and save the largest face
        cropped_face = img[y - offset : y+h + offset , x:x+w]
        cropped_face = cv2.resize(cropped_face, (100,100))
        
        #cv2.imshow("Image Window", img)

        #Predict the name using KNN
        classPredicted = knn(XT,yT, cropped_face.flatten())

        #Name
        namePredicted = nameMap[classPredicted]

        #Display the name and the box
        cv2.putText(img,namePredicted, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255.0),2)

    cv2.imshow("Prediction Window", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


