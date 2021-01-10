import cv2
import os
from collections import defaultdict

files = defaultdict(lambda: [])
classes = defaultdict(lambda: len(classes))

INPUT_FILE_PATH =  "_annotations.csv"
OUTPUT_FILE_PATH = "_custom_train_mosaic.txt"
NAMES_PATH = "Yolov4 Source/custom.names"   

#imagestring
images1s,images2s,images3s,images4s = [],[],[],[]
#actual image
images1n,images2n,images3n,images4n = [],[],[],[]

def loadImages(path = "Yolov4 Source/combined_classes"):
    temp = os.listdir(path)  
    decrease_for_max_images = (len(temp)%4 )  #removes images that can't be combined in another 4-tile
    for i,img in enumerate(temp):  
        if img.endswith(".jpg"): 
            if(i <= (len(temp)) - decrease_for_max_images ):                   
                if(i%4==0) :   
                    images1s.append(img)
                    images1n.append( cv2.imread(f"Yolov4 Source/combined_classes/{(images1s[int(i/4)])}", cv2.IMREAD_UNCHANGED))
                    images1n[0] =resizeimg(images1n[0])               
                if(i%4==1):           
                    images2s.append(img) 
                    images2n.append( cv2.imread(f"Yolov4 Source/combined_classes/{(images2s[int(i/4)])}", cv2.IMREAD_UNCHANGED))
                    images2n[0] =resizeimg(images2n[0])                    
                if(i%4==2 ):           
                    images3s.append(img)
                    images3n.append( cv2.imread(f"Yolov4 Source/combined_classes/{(images3s[int(i/4)])}", cv2.IMREAD_UNCHANGED))
                    images3n[0] =resizeimg(images3n[0])
                if(i%4==3):          
                    images4s.append(img)
                    images4n.append( cv2.imread(f"Yolov4 Source/combined_classes/{(images4s[int(i/4)])}", cv2.IMREAD_UNCHANGED))
                    images4n[0] =resizeimg(images2n[0])
           
            if (i%4 ==3 and i!=0 and i <= (len(temp)) - decrease_for_max_images ):  
                comb_filename = f'{str(images1s[int(i/4.01)])}{str(images2s[int(i/4.01)])}{str(images3s[int(i/4.01)])}{images4s[int(i/4.01)]}'
                comb_file_id = str(int(i/4.01))+"a.jpg"
                im_tile = concat_tile([[images1n[0], images2n[0]],
                                        [images3n[0],images4n[0]]])  
                cv2.imwrite(f'Yolov4 Source/combined_tiled/{comb_file_id}', im_tile)

                #clear memory
                images1n.clear()
                images2n.clear()
                images3n.clear()
                images4n.clear()

                #write combination of filenames code
                with open(INPUT_FILE_PATH, "r") as f:
                    file = f.readlines()
                    for j,line in enumerate(file):
                        if len(line) > 5:
                            if line.split(",")[0].replace(".jpg","") in comb_filename:
                                filename, width, height, class_name, xmin, ymin, xmax, ymax = line.split(',')  
                                #box1  
                                if filename in images1s:                           
                                    width = int(width)
                                    height = int(height)
                                    xmin = (int(xmin)/2) 
                                    xmax = (int(xmax)/2)
                                    ymin = (int(ymin)/2) +256
                                    ymax = (int(ymax)/2) +256
                                    class_num = classes[class_name]
                                    center_x = (xmin + xmax) / 2 / width
                                    center_y = (ymin + ymax) / 2 / height
                                    width = (xmax - xmin) / width
                                    height = (ymax - ymin) / height
                                    files[comb_file_id].append([class_num, center_x, center_y, width, height])
                                #box2   
                                if filename in images2s:                           
                                    width = int(width)
                                    height = int(height)
                                    xmin = (int(xmin)/2 )  +256
                                    xmax = (int(xmax)/2)   +256
                                    ymin = (int(ymin)/2)   +256
                                    ymax = (int(ymax)/2)   +256
                                    class_num = classes[class_name]
                                    center_x = (xmin + xmax) / 2 / width
                                    center_y = (ymin + ymax) / 2 / height
                                    width = (xmax - xmin) / width
                                    height = (ymax - ymin) / height
                                    files[comb_file_id].append([class_num, center_x, center_y, width, height])                                    
                               #box3   
                                if filename in images3s:                           
                                    width = int(width)
                                    height = int(height)
                                    xmin = int(xmin)/2
                                    xmax = int(xmax)/2
                                    ymin = (int(ymin)/2)  
                                    ymax = (int(ymax)/2)  
                                    class_num = classes[class_name]
                                    center_x = (xmin + xmax) / 2 / width
                                    center_y = (ymin + ymax) / 2 / height
                                    width = (xmax - xmin) / width
                                    height = (ymax - ymin) / height
                                    files[comb_file_id].append([class_num, center_x, center_y, width, height])
                                #box4   
                                if filename in images4s:                           
                                    width = int(width)
                                    height = int(height)
                                    xmin = (int(xmin)/2) +256
                                    xmax = (int(xmax)/2) +256
                                    ymin = (int(ymin)/2) 
                                    ymax = (int(ymax)/2) 
                                    class_num = classes[class_name]
                                    center_x = (xmin + xmax) / 2 / width
                                    center_y = (ymin + ymax) / 2 / height
                                    width = (xmax - xmin) / width
                                    height = (ymax - ymin) / height
                                    files[comb_file_id].append([class_num, center_x, center_y, width, height])
                            #print(files[comb_file_id])

     
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def resizeimg(im):
    im1_s = cv2.resize(im, dsize=(256, 256), fx=0.5, fy=0.5)
    return im1_s

loadImages()

with open(OUTPUT_FILE_PATH, "w") as f:
    for filename in files.keys():
        f.write(filename)
        for detected_object in files[filename]:
            f.write(" " + ','.join(map(str, detected_object)))
        f.write('\n')
  
