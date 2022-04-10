import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import os

data_path = 'D:/PYTHON_PROGRAMMING/FACES FOUND/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    Training_data.append(np.asarray(images , dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels , dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_data), np.asarray(Labels))
print("Model Traing Complete")

face_classifier = cv2.CascadeClassifier('C:/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img , size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3 ,5)

    if faces is():
        return img,[]

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h , x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi


def face_extract(img):
    gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gry,1.3,5)

    if faces is():
        return  None
    for(x,y,w,h) in faces:
        crop_face=img[y:y+h, x:x+w]

    return  crop_face


cap = cv2.VideoCapture(0)
count = 0
NotMatched = 0

while True :    
    ret , frame = cap.read()
    image , face = face_detector(frame)

    try:
        face = cv2.cvtColor(face , cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1-(result[1]) / 300))
            display_str = str(confidence) + '% Match'
            cv2.putText(image, display_str, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)


        if confidence>75:
            cv2.putText(image, 'Matched', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper',image)
            count+=1
            NotMatched=0


        if confidence<=75:
            cv2.putText(image, 'Not Matched', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            count = 0
            NotMatched +=1


                # cv2.putText(image, 'You are Imposter', (190, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4)
                # cv2.imshow('Face Cropper', image)
                # t=10
                # while t:
                #     shutdown(image,t)
                #     t-=1



    except:
        cv2.putText(image, 'Face Not Found', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Cropper', image)

    if NotMatched >= 30:
        sount = 0
        if face_extract(frame) is not None:
            sount += 1
            face = cv2.resize(face_extract(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_path = 'D:/PYTHON_PROGRAMMING/IMPOSTER/imposter' + str(sount) + '.jpg'
            cv2.imwrite(file_path, face)
            break

    if cv2.waitKey(1)==13 or count==40:
        break

cap.release()
cv2.destroyAllWindows()

if NotMatched>=30:
    import python_tele_bot
    if python_tele_bot.shut:
        os.system("shutdown /s /t 1")
