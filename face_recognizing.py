import cv2 #handling images
import numpy #array
import os #handling a directories

alg = "haarcascade_frontalface_default.xml"
haar_cascade =  cv2.CascadeClassifier(alg)
datasets = "dataset"
print('Traning...')
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label =id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)
(width,height) =(130,100)

#model = cv2.face.LBPHFaceRecognizer_create()
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

cam = cv2.VideoCapture(0)
count = 0

while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 5)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,255,0), 2)
        faceonly = grayImg[y:y+h,x:x+w]
        resizeImg = cv2.resize(faceonly,(width,height))
        
        prediction = model.predict(resizeImg)
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 3)
        if prediction[1]<800:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0), 2)
            print (names[prediction[0]])
            count = 0
        else:
            count+=1
            cv2.putText(img,'unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0), 2)
            if(count>100):
                print("Unknown Person")
                cv2.imwrite("unknown.jpg",img)
                count=0
    cv2.imshow("Face_recognization",img)
    key = cv2.waitKey(30)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
