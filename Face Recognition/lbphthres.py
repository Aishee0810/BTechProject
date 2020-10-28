#names = ["Aishee","Akanksha","Arkaprabha","Parnavi","Ritodeep","Rohit","Sayan","Sukrita"]

import pandas as pd
name_data = pd.read_csv("HOG2/name_database.csv")
names = list(name_data["Names"])


from tqdm import tqdm

import os

import numpy as np

import cv2                   
import os.path

import re

import os

from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
                
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

def sort_int(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def test_data(DIR):
    xtest=[]
    paths=[]
    for img in tqdm(sort_int(os.listdir(DIR))):
        path = os.path.join(DIR,img)
        paths.append(path)
        img = cv2.imread(path,0)
        img = cv2.resize(img, (150,150))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        xtest.append(np.array(img))
        
    return xtest,paths

def predict(path):
#    pred=[]

#    path = 'HOG2/test'
    x_test_k,paths_k =test_data(path+"/known")
    x_test_uk,paths_uk =test_data(path+"/unknown")
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#    face_recognizer.read("lbph.json")
    face_recognizer.read("lbph_15.json")
#    face_recognizer.update()
    
#    for i in range(len(x_test)):
#        pred.append(face_recognizer.predict(x_test[i]))
#        
#    pred_name = [names[pred[i][0]] for i in range(len(pred))]
    
    threshold = np.arange(69,76.5,.5)

    correct_list = []
    for thres in threshold:
        c = 0
        for i in range(len(x_test_k)):
            prediction = face_recognizer.predict(x_test_k[i])
            if prediction[1] > thres:
                continue
            else:
                c += 1
                
        for i in range(len(x_test_uk)):
            prediction = face_recognizer.predict(x_test_uk[i])
            if prediction[1] > thres:
                c += 1
            else:
                continue
            
        correct_list.append(c)
    
    print(correct_list)
    plt.plot(correct_list)
    
    index = []
    m = max(correct_list)
    for i in range(0,len(correct_list)):
        if correct_list[i] == m:
            index.append(i)
            
    print(threshold[index[-1]])

    
if __name__ == '__main__':
    path = "threshold_test"
    predict(path)