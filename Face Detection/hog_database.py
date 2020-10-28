import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import dlib
import time
# data visualisation and manipulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from PIL import Image
import pandas as pd
 
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

import cv2                  
from tqdm import tqdm
import os
import os.path


def face_detecthog(image,sname,iname):
    hog_face_detector = dlib.get_frontal_face_detector()
    # start = time.time()

      # apply face detection (hog)
    faces_hog = hog_face_detector(image, 1)
    # end = time.time()
    # print("Execution Time (in seconds) :")
    # print("HOG : ", format(end - start, '.2f'))
  
    i=0
    #loop over detected faces
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

  # draw box over face
  
        if w < 80 or h < 80:
            continue
      
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_color = image[y:y+h, x:x+h]
        # print(roi_color)
        # plt.imshow(roi_color)
        # plt.show()
        
        path1 = "HOG_Data"
        if not os.path.exists(path1):
            os.mkdir(path1)       
        path = path1 + "/" + sname
        if not os.path.exists(path):
            os.mkdir(path)
        
        print(path)
        cv2.imwrite(path+'/'+iname,roi_color)
        i+=1
      
     
def hog_folder(DIR):
    print(DIR)
    for s_name in tqdm(os.listdir(DIR)):
        path1 = DIR+"/"+s_name
        for i_name in tqdm(os.listdir(path1)):
            path = path1+"/"+i_name
            print(path)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_detecthog(img,s_name,i_name)
        print(s_name+" complete")

if __name__ == '__main__':
    path = "Original"       
    hog_folder(path)