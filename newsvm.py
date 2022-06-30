from os import walk
import os
import csv

import numpy as array
from sklearn.svm import SVC
# loading the iris dataset
from joblib import dump, load
import winsound
data_set=None
X=y=None
from numpy import array

data_set=None
X=y=None
# from myknn import *
from glcm import glfeaturess,glfeature
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split






def loadfilename():

    list_of_lists = []
    with open('data.csv') as f:
        for line in f:
            inner_list = [line.strip() for line in line.split(',')]
            list_of_lists.append(inner_list)


    i=0 
    xx=[]
    yy=[]
    for lis in list_of_lists:
        # print("ppp",lis)
        xxx=[]

        xxx.append(float(lis[0]))
        xxx.append(float(lis[1]))
        xxx.append(float(lis[2]))
        xxx.append(float(lis[3]))
        xxx.append(float(lis[4]))
        xxx.append(float(lis[5]))
        
        xx.append(xxx)
        # print(xx)
        yy.append(lis[6])
        # print(lis[18])
        # i=i+1
        # if i%10==0:
        #     x1.append(xxx)
        #     y1.append(lis[6])

    X = array( xx )
    y = yy



    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    svm_model_linear = SVC(kernel='linear', C=1).fit(X, y)

    dump(svm_model_linear, 'filename.joblib')
    svm_predictions = svm_model_linear.predict(X_test)
    print(len(X_test),X_test)
    print(svm_predictions)
# model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    print("accuracy"+str(accuracy))
    # try:
    #     rys="insert into `accuracy` VALUES('svm',%s)"
    #     vals=(str(accuracy),)
    #     insert(rys,vals)
    # except Exception as e:
    #     print ("err")
# creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)





def train():
    data=[]
    samples=[]
    names=[]
    for dir,d_path,filnames in walk("static/dataset/fire_images"):
      for file in filnames:
          sample=glfeaturess(os.path.join(dir,file),1)
          samples.append(sample)
    for dir,d_path,filnames in walk(r"static/dataset/non_fire_images"):
          for file in filnames:
            sample=glfeaturess(os.path.join(dir,file),2)
            samples.append(sample)
    



    # print(samples)

    


#train() 
    
#loadfilename()



def predictsvm(fn):
    xx=[]
    glt = fn
    print(glt)
    feat=glfeature(fn)
    xx.append(feat)
    print(xx)
    svc = load('filename.joblib')
    p = svc.predict(xx)
    print(p)
    if p=='1':
        # frequency is set to 500Hz
                freq = 500
                # duration is set to 100 milliseconds           
                dur = 800
                winsound.Beep(freq, dur)
    return p

# val=['162.60385858485398', '7.538391924817261', '0.1981536882892652', '0.0004695496110158305', '0.021669093451638224', '0.982733227442188']
# predictsvm("static/dataset/fire_images/Datacluster Fire and Smoke Sample (1).jpg")





