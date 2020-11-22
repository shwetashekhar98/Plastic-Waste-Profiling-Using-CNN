# import the necessary packages
from nms import non_max_suppression_fast
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
from tkinter import filedialog
from tkinter import *
import tkinter as tk
import sqlite3 
import time
import os
from pyimagesearch.helpers import sliding_window

class deep:
    def __init__(self,root):
        self.l1= Button(root,text = "Select File",command = self.file_open,fg="Black",padx=13,pady=6,bd=4,width=25)
        self.l1.pack()
        self.l2= Button(root,text = "test",command = self.test,fg="Black",padx=13,pady=6,bd=4,width=25)
        self.l2.pack()

    # load the image
    def file_open(self):
        ob.filename =  filedialog.askopenfilenames(initialdir = "C:/Users/Sri Satya Sai/Pictures/deepblue/test/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

    def test(self):
        for i in range (0,len(ob.filename)):
            def insertdb(f,cl,pa,count):
                print(f,cl,pa,count)
                localtime = time.asctime( time.localtime(time.time()) )
                conn=sqlite3.connect("model_stats.db")
                c=conn.cursor()
                #c.execute("drop table logs")
                #c.execute("create table logs (File_Name varchar(20),Prediction varchar(10), Accuracy float, Count int, Time varchar(50))")
                c.execute("insert into logs values('%s','%s','%.2f','%d','%s')" %(f,cl,pa*100,count,localtime))      
                conn.commit()

            name = ob.filename[i]
            img = cv2.imread(name)
            img = cv2.resize(img, (450,350))
            (winW, winH) = (160, 160)
            xa=[]
            ya=[]
            lab=[]
            b=[]
            acc=0

            for (x, y, window) in sliding_window(img, stepSize=50, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                # pre-process the image for classification
                image = cv2.resize(window, (28, 28))
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)        
                # load the trained convolutional neural network
                # print("[IN`FO] loading network...")
                model = load_model("C:/Users/Sri Satya Sai/Pictures/deepblue/predict.model")
                # classify the input image
                (not_sprite, sprite) = model.predict(image)[0]
                # build the label
                label = "sprite" if sprite > not_sprite else "not sprite"
                if label == "sprite":
                    l = label
                else:
                    nl = label
                proba = sprite if sprite > not_sprite else not_sprite
                label = "{}: {:.2f}%".format(label, proba * 100)
                if proba == sprite and sprite>=0.95:
                    xa.append(x)
                    ya.append(y)
                    lab.append(label)
                    acc = acc+sprite

            if not xa or not ya:
                cv2.putText(img, nl, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.7, (0, 250, 0), 2)
                #img = imutils.resize(img, width = 400)
                cv2.imshow("Output", img)
                cv2.waitKey(0)
                insertdb(os.path.basename(name),nl,proba,0) 
            else:
                for i,j,k in zip(xa,ya,lab):
                    b.append((i,j,i+winW,j+winH))
                box = [np.array(b)]
                acc=acc/len(lab)
                for (boundingBoxes) in box:
                    #print("%d initial bounding boxes" % (len(boundingBoxes)))

                    pick = non_max_suppression_fast(boundingBoxes, 0.3)
                    print("No. of detections: %d " % (len(pick)))
                    print(acc)
                    for startX, startY, endX, endY in pick:
                        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)   
                        cv2.putText(img, l, (startX, startY-5),  cv2.FONT_HERSHEY_SIMPLEX,	0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Output", img)
                    cv2.waitKey(0)

                insertdb(os.path.basename(name),l,acc,len(pick))            
root = Tk()
ob=deep(root)
root.mainloop()