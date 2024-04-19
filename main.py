from flask import Flask, render_template,request
import HandTrackingModule as htm#this is class we create object from this class
import math
import numpy as np
import cv2
import mediapipe as mp
import time
from comtypes import CLSCTX_ALL
from time import sleep
import pyautogui  # Used for mouse control
from pynput.keyboard import Controller
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


app = Flask(__name__)

#first page
@app.route('/') #/ this froward slash site run the site first
def index():
   return render_template('index.html') 


@app.route('/Explore') 
def Explore():
     return render_template('Explore.html')



@app.route('/about') 
def about():
     return render_template('about.html')

@app.route('/contact') 
def contact():  
     return render_template('contact.html')

@app.route('/Action_Page') 
def Action_Page():  
     return render_template('Action_Page.html')


@app.route('/Virtual_Volume') 
def Virtual_Volume():  

   
           #camera width and height
    wCam , hCam = 640,480

#######################################

    cap = cv2.VideoCapture(0)
    #prop id
    cap.set(3,wCam)
    cap.set(4,hCam)
    pTime = 0


    detector = htm.handDetector(detectionCon=0.7)


    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()#then we get ranger -65 to 0
#volume.SetMasterVolumeLevel(0, None)#0 = 100% volume -5 = 72% 
    minVol = volRange[0]
    maxVol = volRange[1]
#next thing to covert volume ranges
    vol = 0 #it show error name error that's why we define here to obally access
    volBar = 400 # same here also
    volPer = 0
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
   
        lmlist = detector.findPosition(img,draw = False)#draw false beacase we already drawing it
        if len(lmlist) != 0:
     #print(lmlist[4],lmlist[8])#getting values for particular
#now we need value for thumb tip and 8 for index finger tip
     
            x1,y1 = lmlist[4][1],lmlist[4][2]#4id 1 element and 2 element
            x2,y2 = lmlist[8][1],lmlist[8][2]
            cx,cy = (x1+x2)//2 , (y1+y2) // 2
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
     #creating lines betweeen points
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
     #now we have to find the length of the line then we can change volume based on that  
     
            length = math.hypot(x2-x1,y2-y1)#gives us length
      #print(length)#gives 342.12,233.2323 max is 3 hund around something
                  #min is 50 around
      

      # Hand Range 50 - 300
      # Covert in Volume Range
      # Volume Range is -65 - 0

            vol  = np.interp(length,[50,300],[minVol,maxVol])
            volBar = np.interp(length,[50,300],[400,150])
            volPer = np.interp(length,[50,300],[0,100])
      #print(int(length),vol)
      #we convert this now we can send it to master volume 
            volume.SetMasterVolumeLevel(vol, None)


            if length<50:
                cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
        #change volume based on length
        #we pycaw library develop by andrew to change volume licensed by mit
          

    #create volume bar width = 85 - 50 = 35
        cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
    #widht is same but heigth is different
    #not write vol directly beacoz bar get out of image
    #when volume is 0 then 400 when vol is max then it should 150
        cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)

    #write percentage so again we covert percentage conversion above
        cv2.putText(img,f'{int(volPer)} %',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    





        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)


        cv2.imshow("Img",img)
        cv2.waitKey(1)
        if cv2.waitKey(2) & 0xFF == ord('q'):
          cap.release()
          cv2.destroyAllWindows()
          return render_template('Action_Page.html')
     

@app.route('/Virtual_Cursor') 
def Virtual_Cursor():  
 

 

#####################
 wCam, hCam = 640, 480
 frameR = 100  # Frame Reduction
 smoothening = 10
#####################

 pTime = 0
 plocX, plocY = 0, 0
 clocX, clocY = 0, 0

 cap = cv2.VideoCapture(0)
 cap.set(3, wCam)
 cap.set(4, hCam)

 mpHands = mp.solutions.hands
 hands = mpHands.Hands()
 mpDraw = mp.solutions.drawing_utils

 wScr, hScr = pyautogui.size()

 while True:
     success, img = cap.read()

     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     results = hands.process(imgRGB)

     lmList = []
     if results.multi_hand_landmarks:
         for handLms in results.multi_hand_landmarks:
             for id, lm in enumerate(handLms.landmark):
                 h, w, c = img.shape
                 cx, cy = int(lm.x * w), int(lm.y * h)
                 lmList.append([id, cx, cy])
             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

     if len(lmList) != 0:
         x1, y1 = lmList[8][1:]
         x2, y2 = lmList[12][1:]

         fingers = [0, 0]
         if y1 < lmList[6][2]:
             fingers[0] = 1
         if y2 < lmList[10][2]:
             fingers[1] = 1

         cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

         if fingers[1] == 1 and fingers[0] == 0:
             x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
             y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

             clocX = plocX + (x3 - plocX) / smoothening
             clocY = plocY + (y3 - plocY) / smoothening
             pyautogui.moveTo(wScr - clocX, clocY)
             cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
             plocX, plocY = clocX, clocY

         if fingers[1] == 1 and fingers[0] == 1:
             length = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
             print(length)

             if length < 40:
                 cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                 pyautogui.click()

     cTime = time.time()
     fps = 1 / (cTime - pTime)
     pTime = cTime
     cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

     cv2.imshow("Image", img)
     if cv2.waitKey(1) & 0xFF == ord('q'):
          cap.release()
          cv2.destroyAllWindows()
          return render_template('Action_Page.html')
     
     

         


        


@app.route('/Virtual_Keyboard') 
def Virtual_Keyboard():  
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpdraw = mp.solutions.drawing_utils

    keyboard = Controller()

    cap = cv2.VideoCapture(0)
    cap.set(2,150)

    text = ""
    tx = ""

    class Button():
        def __init__(self, pos, text, size=[70, 70]):
            self.pos = pos
            self.size = size
            self.text = text

    keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P","CL"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";","SP"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/","APR"]]

    keys1 = [["q", "w", "e", "r", "t", "y", "u", "i", "o", "p","CL"],
         ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";","SP"],
         ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/","APR"]]

    def drawAll(img, buttonList):
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            cv2.rectangle(img, button.pos, (x + w, y + h), (96, 96, 96), cv2.FILLED)
            cv2.putText(img, button.text, (x + 10, y + 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        return img

    buttonList = []
    buttonList1 = []
    list = []


    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([80 * j + 10, 80 * i + 10], key))

    for i in range(len(keys1)):
        for j, key in enumerate(keys1[i]):
            buttonList1.append(Button([80 * j + 10, 80 * i + 10], key))

    app = 0      
    delay = 0

    def calculate_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2) 

    while True:
        success, frame = cap.read()

        frame = cv2.resize(frame, (1000, 580))
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        lanmark = []

        if app == 0:
            frame = drawAll(frame, buttonList) 
            list = buttonList
            r = "up"

        if app == 1:
            frame = drawAll(frame, buttonList1) 
            list = buttonList1 
            r = "down"
        if results.multi_hand_landmarks:
            for hn in results.multi_hand_landmarks:
                for id, lm in enumerate(hn.landmark):
                    hl, wl, cl = frame.shape
                    cx, cy = int(lm.x * wl), int(lm.y * hl)     
                    lanmark.append([id, cx, cy]) 
        if lanmark:
            try:
                x5, y5 = lanmark[5][1], lanmark[5][2]
                x17, y17 = lanmark[17][1], lanmark[17][2]
                dis = calculate_distance(x5, y5, x17, y17)
                A, B, C = coff
                distanceCM = A * dis ** 2 + B * dis + C

                if 20 < distanceCM < 50:
                    x, y = lanmark[8][1], lanmark[8][2]
                    x2, y2 = lanmark[6][1], lanmark[6][2]
                    x3, y3 = lanmark[12][1], lanmark[12][2]
                    cv2.circle(frame, (x, y), 20, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x3, y3), 20, (255, 0, 255), cv2.FILLED)



                    if y2 > y:
                        for button in list:
                            xb, yb = button.pos
                            wb, hb = button.size


                            if xb < x < xb + wb and yb < y < yb + hb:
                                cv2.rectangle(frame, (xb - 5, yb - 5), (xb + wb + 5, yb + hb + 5), (160, 160, 160), cv2.FILLED)
                                cv2.putText(frame, button.text, (xb + 20, yb + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                                dis = calculate_distance(x, y, x3, y3)

                                if dis < 50 and delay == 0:
                                    k = button.text
                                    cv2.rectangle(frame, (xb - 5, yb - 5), (xb + wb + 5, yb + hb + 5), (255, 255, 255), cv2.FILLED)
                                    cv2.putText(frame, k, (xb + 20, yb + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

                                    if k == "SP":
                                        tx = ' '  
                                        text += tx
                                        keyboard.press(tx)
                                    elif k == "CL":
                                        tx = text[:-1]
                                        text = ""
                                        text += tx
                                        keyboard.press('\b')
                                    elif k == "APR" and r == "up":
                                        app = 1
                                    elif k == "APR" and r == "down":
                                        app = 0
                                    else:
                                        text += k
                                        keyboard.press(k)
                                    delay = 1       
            except:
                pass

        if delay != 0:
                delay += 1
        if delay > 10:
                delay = 0

        cv2.rectangle(frame, (20, 250), (850, 400), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (30, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.imshow('virtual keyboard', frame )

        if cv2.waitKey(1) & 0xFF == ord('q'):
          cap.release()
          cv2.destroyAllWindows()
          return render_template('Action_Page.html')


if __name__== "__main__":
    app.run(debug=True,port=5001)

