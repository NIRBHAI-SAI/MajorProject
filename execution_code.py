def facerecognition():

    import os
    import cv2
    import numpy as np
    from os import listdir
    from os.path import isfile, join
    from tts.announce import announcer
    import time
    from dataset_creation import data_creation



    # hel = "hello uday"
    # myobj = gTTS(text = hel, lang = 'en', slow=False)
    # myobj.save("hel.mp3")
    # os.system("hel.mp3")

    data_path =  'D:\\sem 8\\majorproject 2\\currency_old\\faces\\user\\'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    # print(onlyfiles)

    Training_Data, Labels = [], []
    label_name = {}


    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.split(image_path)[-1].split('.')[1])
        Training_Data.append(np.asarray(images, dtype = np.uint8))
        label_name[id]=os.path.split(image_path)[-1].split('.')[0]
        Labels.append(id)

    # print(label_name)
    Labels = np.asarray(Labels, dtype = np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained successfully")


    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_detector(img, size =0.5):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces == ():
            return img, []

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
        return img, roi

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    begin = time.time()
    count = 0
    while ret:


        count= count+1
        #print(count)

        ret, frame = cap.read()

        image, face = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            results = model.predict(face)
            #print(results[1])
            #cv2.imshow('Face Recognition', image )

            confidence = int(100*(1-(results[1])/400))
            # if results[1] <500:
            #     confidence = int(100*(1-(results[1])/400))
            print(confidence,results[0])
            if confidence >= 90:
                display_string = str(confidence) + '% Confident it is '+ label_name[results[0]]
                cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
                #print(display_string)
                print(label_name[results[0]] + " is detected ")
                announcer(label_name[results[0]] + " is detected ")
                break

            else:
                if count >= 7:
                    print("unknown face")
                    cv2.putText(image, "unknown face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    announcer(" unknown face is detected")
                    data_creation()
                    break


                #cv2.imshow('unrecognition', image )


            #$cv2.imshow('Face', image )
        except:
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            # cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            #cv2.imshow('Face Recognition', image )
            #print("NO Face found")
            pass
        # cv2.imshow('face recognition', image )
        # time.sleep(1)
        end = time.time()
        # print(end-begin)

        if end - begin > 10:
            announcer("No face found")
            break


        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()



