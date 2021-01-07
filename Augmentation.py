from collections import defaultdict
from numpy.lib.npyio import load
from numpy.lib.type_check import imag
import numpy as np
import cv2
import random
import os

#this script adds the new images to the regular export folder
INSTANCES_PATH = "export/_annotations.csv"  
NAMES_PATH = "export/_annotations_comb.csv"                       

files = defaultdict(lambda: [])
classes = defaultdict(lambda: len(classes))
underrep_unique = []
imagesstring = []
images = []
d = []

'''
get all underrepresented images and stor in array
'''
with open(INSTANCES_PATH) as f:
    f.readline() 
    for line in f:
        if len(line) > 5:            
            if (  ( "pedestrian" in line 
                or "trafficLight-Red" in line 
                or "trafficLight-Green" in line 
                or "truck" in line 
                or "biker" in line) and
                (not "trafficLight-RedLeft" in line
                and not "trafficLight-GreenLeft" in line)):
                filename, width, height, class_name, xmin, ymin, xmax, ymax = line.split(',')            
                if (filename[:26] not in underrep_unique):
                    underrep_unique.append(filename[:26])
                    d.append(filename)

'''
the raw images in export file
''' 
def loadImages(path = "/export"):
    #return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]  
    temp = os.listdir(path)  
    for i in temp:       
        if i.endswith("jpg"):
            if i in d:
                imagesstring.append(i)
loadImages()

'''
function that adds noise to images
'''
def noisy(noise_typ,image):
   rng = random.randrange(-2,2)
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean =  0
      var = 1.3
      sigma = var**2
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = (0.8 *image)  + 2 * rng *gauss
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + (0.2 *image * gauss)
      return noisy

'''
make folder of all the underrepresented images
'''
for file in imagesstring:    
    images.append( cv2.imread(f"Yolov4 Source/export/{file}", cv2.IMREAD_UNCHANGED))
for i,image in enumerate(images):
    image_rg = image * np.array([0,1,1]) #b g r
    imagesstring[i] = imagesstring[i].replace(".jpg","")
    cv2.imwrite(str(f"/export/{imagesstring[i]}_rg.jpg"),image_rg)
    image_gb = noisy("speckle",image)
    image_gb = noisy("gauss",image_gb)      
    imagesstring[i] = imagesstring[i].replace(".jpg","")    
    cv2.imwrite(str(f"/export/{imagesstring[i]}_gblur.jpg"),image_gb)




