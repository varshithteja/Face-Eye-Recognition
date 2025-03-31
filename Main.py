'''from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import cv2
import face_recognition
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import os
import socket
import pickle

main = tkinter.Tk()
main.title("Federated Learning Based Face and Eye Blink Recognition")
main.geometry("1300x1200")

global filename, user, names, encodings, faceCascade, eyeblinks, eye_cam, counter, total

def loadModel():
    global names, encodings, faceCascade, eyeblinks
    if os.path.exists("model/encoding.npy"):
        encodings = np.load("model/encoding.npy")
        names = np.load("model/names.npy")        
    else:
        encodings = []
        names = []        
    if os.path.exists("model/blinks.npy"):
        eyeblinks = np.load("model/blinks.npy")
    else:
        eyeblinks = []
    cascPath = "model/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    print(eyeblinks)
    print(names)
loadModel()

def saveFace():
    global names, encodings
    encodings = np.asarray(encodings)
    names = np.asarray(names)
    np.save("model/encoding", encodings)
    np.save("model/names", names)

def faceDetection():
    global user, names, encodings, faceCascade, eyeblinks
    text.delete('1.0', END)
    text.insert(END,'Dataset Path Loaded\n\n')
    user = tf1.get()
    tf1.delete(0, END)
    text.insert(END,"Username entered as "+user+"\n")
    text.update_idletasks()
    done = False
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             faces = faceCascade.detectMultiScale(gray,1.3,5)
             print("Found {0} faces!".format(len(faces)))
             if len(faces) > 0:
                 for (x, y, w, h) in faces:
                     frame = cv2.resize(frame, (600, 600))
                     cv2.imwrite("test.png", frame)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             cv2.imshow("Output", frame)
             if cv2.waitKey(15) & 0xFF == ord('q'):
                 image = face_recognition.load_image_file("test.png")
                 encoding = face_recognition.face_encodings(image)
                 print("encoding "+str(encoding))
                 if len(encoding) > 0 and user not in names:
                     encoding = encoding[0]
                     if len(encodings) == 0:
                         encodings.append(encoding)
                         names.append(user)
                     else:
                         encodings = encodings.tolist()
                         names = names.tolist()
                         encodings.append(encoding)
                         names.append(user)
                     saveFace()
                     done = True
                     text.insert(END,"Face Registration Completed\n")
                     text.update_idletasks()
                     break
                 else:
                     text.insert(END,"Face not detected please retry\n")
                     text.update_idletasks()
             if done:
                 break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def saveBlink(total, user):
    global eyeblinks
    if len(eyeblinks) == 0:
        eyeblinks.append([user, total])
    else:
        eyeblinks = eyeblinks.tolist()
        eyeblinks.append([user, total])
    eyeblinks = np.asarray(eyeblinks)
    np.save("model/blinks", eyeblinks)

def eyeBlink():
    global eyeblinks, eye_cam, counter, total, user
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    counter = 0
    total = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < EYE_AR_THRESH:
                    counter += 1
                else:
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        total += 1
                    counter = 0
                cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                saveBlink(total, user)
                text.insert(END,"Eye Blinks Registration Completed\n")
                text.update_idletasks()
                break
    cap.release()        
    cv2.destroyAllWindows()
    
def updateModel():
    text.delete('1.0', END)
    global names, encodings, faceCascade, eyeblinks
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 2222))
    data = []
    data.append(encodings)
    data.append(names)
    data.append(eyeblinks)
    data = pickle.dumps(data)
    s.sendall(data)
    data = s.recv(100)
    data = data.decode()
    text.insert(END,"Server Response : "+data+"\n\n")
    s.close()

def blinkAuth():
    global eyeblinks, eye_cam, counter, total, success, user
    success = False
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    counter = 0
    total = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < EYE_AR_THRESH:
                    counter += 1
                else:
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        total += 1
                        for k in range(len(eyeblinks)):
                            eb = eyeblinks[k]
                            #print(str(eb)+" "+str(total)+" "+user)
                            if eb[0] == user and eb[1] == str(total):
                                success = True
                                text.insert(END,"Eye Blinks Pattern Authenticated as : "+eb[0]+"\n")
                                text.update_idletasks()
                    counter = 0
                cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if success == True:
                break
    cap.release()        
    cv2.destroyAllWindows()
    
    
def authentication():
    global names, encodings, faceCascade, eyeblinks, user
    text.delete('1.0', END)
    done = False
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             faces = faceCascade.detectMultiScale(gray,1.3,5)
             print("Found {0} faces!".format(len(faces)))
             if len(faces) > 0:
                 for (x, y, w, h) in faces:
                     small_frame = cv2.resize(frame, (600, 600))
                     cv2.imwrite("test.png", small_frame)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             cv2.imshow("Frame", frame)
             if cv2.waitKey(5) & 0xFF == ord('q'):
                 img = cv2.imread("test.png")
                 small_frame = cv2.resize(img, (600, 600))
                 rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB color space
                 face_locations = face_recognition.face_locations(rgb_small_frame)  # Locate faces in the frame
                 face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Encode faces in the frame
                 for face_encoding in face_encodings:
                     matches = face_recognition.compare_faces(encodings, face_encoding)  # Compare face encodings
                     face_distance = face_recognition.face_distance(encodings, face_encoding)  # Calculate face distance
                     best_match_index = np.argmin(face_distance)  # Get the index of the best match
                     print(best_match_index)
                     if matches[best_match_index]:  # If the face is a match
                         name = names[best_match_index]  # Get the corresponding name
                         user = name
                         text.insert(END,"Face Recognized As : "+name+"\n")
                         text.update_idletasks()
                         done = True
                         break
                 if done:
                     break
    cap.release()        
    cv2.destroyAllWindows()    

    
font = ('Aeial Black', 18, 'bold')
title = Label(main, text='Federated Learning Based Face and Eye Blink Recognition')
title.config(bg='#A9DCE3', fg='#7689DE')  
title.config(font=font)           
title.config(height=3, width=150)       
title.place(x=0,y=5)

l1 = Label(main, text='Username')
l1.config(font=font)
l1.place(x=700,y=100)

tf1 = Entry(main,width=25)
tf1.config(font=font)
tf1.place(x=850,y=100)

font1 = ('times', 13, 'bold')
faceDetection = Button(main, text="Face Detection & Registration", command=faceDetection)
faceDetection.place(x=700,y=150)
faceDetection.config(font=font1)  

eyeblinkButton = Button(main, text="Eyeblink & Local Training", command=eyeBlink)
eyeblinkButton.place(x=700,y=200)
eyeblinkButton.config(font=font1)

updateButton = Button(main, text="Federated Update Model to Server", command=updateModel)
updateButton.place(x=700,y=250)
updateButton.config(font=font1) 

authButton = Button(main, text="Face Authentication", command=authentication)
authButton.place(x=700,y=300)
authButton.config(font=font1)

eyeauthButton = Button(main, text="Eye Blink Authentication", command=blinkAuth)
eyeauthButton.place(x=700,y=350)
eyeauthButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='#7689DE')
main.mainloop()
'''

from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import cv2
import face_recognition
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import os
import socket
import pickle

main = tkinter.Tk()
main.title("Federated Learning Based Face and Eye Blink Recognition")
main.geometry("1300x1200")

global filename, user, names, encodings, faceCascade, eyeblinks, eye_cam, counter, total

def loadModel():
    global names, encodings, faceCascade, eyeblinks
    if os.path.exists("model/encoding.npy"):
        encodings = np.load("model/encoding.npy")
        names = np.load("model/names.npy")        
    else:
        encodings = []
        names = []        
    if os.path.exists("model/blinks.npy"):
        eyeblinks = np.load("model/blinks.npy")
    else:
        eyeblinks = []
    cascPath = "model/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    print(eyeblinks)
    print(names)
loadModel()

def saveFace():
    global names, encodings
    encodings = np.asarray(encodings)
    names = np.asarray(names)
    np.save("model/encoding", encodings)
    np.save("model/names", names)

def faceDetection():    
    global user, names, encodings, faceCascade, eyeblinks
    text.delete('1.0', END)
    text.insert(END,'Dataset Path Loaded\n\n')
    user = tf1.get()
    tf1.delete(0, END)
    text.insert(END,"Username entered as "+user+"\n")
    text.update_idletasks()
    done = False
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             faces = faceCascade.detectMultiScale(gray,1.3,5)
             print("Found {0} faces!".format(len(faces)))
             if len(faces) > 0:
                 for (x, y, w, h) in faces:
                     frame = cv2.resize(frame, (600, 600))
                     cv2.imwrite("test.png", frame)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             cv2.imshow("Output", frame)
             if cv2.waitKey(15) & 0xFF == ord('q'):
                 image = face_recognition.load_image_file("test.png")
                 encoding = face_recognition.face_encodings(image)
                 print("encoding "+str(encoding))
                 if len(encoding) > 0 and user not in names:
                     encoding = encoding[0]
                     if len(encodings) == 0:
                         encodings.append(encoding)
                         names.append(user)
                     else:
                         encodings = encodings.tolist()
                         names = names.tolist()
                         encodings.append(encoding)
                         names.append(user)
                     saveFace()
                     done = True
                     text.insert(END,"Face Registration Completed\n")
                     text.update_idletasks()
                     break
                 else:
                     text.insert(END,"Face not detected please retry\n")
                     text.update_idletasks()
             if done:
                 break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def saveBlink(total, user):
    global eyeblinks
    if len(eyeblinks) == 0:
        eyeblinks.append([user, total])
    else:
        eyeblinks = eyeblinks.tolist()
        eyeblinks.append([user, total])
    eyeblinks = np.asarray(eyeblinks)
    np.save("model/blinks", eyeblinks)

def eyeBlink():

    
    global eyeblinks, eye_cam, counter, total, user
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    counter = 0
    total = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < EYE_AR_THRESH:
                    counter += 1
                else:
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        total += 1
                    counter = 0
                cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                saveBlink(total, user)
                text.insert(END,"Eye Blinks Registration Completed\n")
                text.update_idletasks()
                break
    cap.release()        
    cv2.destroyAllWindows()
    
def updateModel():
    text.delete('1.0', END)
    global names, encodings, faceCascade, eyeblinks
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 2222))
    data = []
    data.append(encodings)
    data.append(names)
    data.append(eyeblinks)
    data = pickle.dumps(data)
    s.sendall(data)
    data = s.recv(100)
    data = data.decode()
    text.insert(END,"Server Response : "+data+"\n\n")
    s.close()

def blinkAuth():
    global eyeblinks, eye_cam, counter, total, success, user
    success = False
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    counter = 0
    total = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < EYE_AR_THRESH:
                    counter += 1
                else:
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        total += 1
                        for k in range(len(eyeblinks)):
                            eb = eyeblinks[k]
                            #print(str(eb)+" "+str(total)+" "+user)
                            if eb[0] == user and eb[1] == str(total):
                                success = True
                                text.insert(END,"Eye Blinks Pattern Authenticated as : "+eb[0]+"\n")
                                text.update_idletasks()
                    counter = 0
                cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if success == True:
                break
    cap.release()        
    cv2.destroyAllWindows()
    
    
def authentication():
    global names, encodings, faceCascade, eyeblinks, user
    text.delete('1.0', END)
    done = False
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             faces = faceCascade.detectMultiScale(gray,1.3,5)
             print("Found {0} faces!".format(len(faces)))
             if len(faces) > 0:
                 for (x, y, w, h) in faces:
                     small_frame = cv2.resize(frame, (600, 600))
                     cv2.imwrite("test.png", small_frame)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             cv2.imshow("Frame", frame)
             if cv2.waitKey(5) & 0xFF == ord('q'):
                 img = cv2.imread("test.png")
                 small_frame = cv2.resize(img, (600, 600))
                 rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB color space
                 face_locations = face_recognition.face_locations(rgb_small_frame)  # Locate faces in the frame
                 face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Encode faces in the frame
                 for face_encoding in face_encodings:
                     matches = face_recognition.compare_faces(encodings, face_encoding)  # Compare face encodings
                     face_distance = face_recognition.face_distance(encodings, face_encoding)  # Calculate face distance
                     best_match_index = np.argmin(face_distance)  # Get the index of the best match
                     print(best_match_index)
                     if matches[best_match_index]:  # If the face is a match
                         name = names[best_match_index]  # Get the corresponding name
                         user = name
                         text.insert(END,"Face Recognized As : "+name+"\n")
                         text.update_idletasks()
                         done = True
                         break
                 if done:
                     break
    cap.release()        
    cv2.destroyAllWindows()    

    
font = ('Aeial Black', 16, 'bold')
title = Label(main, text='Federated Learning Based Face and Eye Blink Recognition')
title.config(bg='#A9DCE3', fg='#7689DE')  
title.config(font=font)      
title.config(height=3, width=120)   
title.pack(pady=0)    
title.place(x=0,y=5)

l1 = Label(main, text='USERNAME')
l1.config(bg="azure")
l1.config(font=font)
l1.place(x=700,y=130)
font1 = ('times', 13, 'italic')
tf1 = Entry(main,width=25,bg="ghostwhite", fg="red4")
tf1.config(font=font)
tf1.place(x=850,y=130)


faceDetection = Button(main, bg="black", fg="white", height="2", width="35", text="Face Detection & Registration", command=faceDetection)
faceDetection.place(x=820,y=200)
faceDetection.config(font=font1)  

eyeblinkButton = Button(main, fg="black", bg="white", height="2", width="35", text="Eyeblink & Local Training", command=eyeBlink)
eyeblinkButton.place(x=820,y=270)
eyeblinkButton.config(font=font1)

updateButton = Button(main, bg="black", fg="white",height="2", width="35", text="Federated Update Model to Server", command=updateModel)
updateButton.place(x=820,y=340)
updateButton.config(font=font1) 

authButton = Button(main, fg="black", bg="white",height="2", width="35", text="Face Authentication", command=authentication)
authButton.place(x=820,y=410)
authButton.config(font=font1)

eyeauthButton = Button(main, bg="black", fg="white",height="2", width="35", text="Eye Blink Authentication", command=blinkAuth)
eyeauthButton.place(x=820,y=480)
eyeauthButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='#7689DE')
main.mainloop()
