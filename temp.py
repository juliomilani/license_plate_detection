import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = 'data/UFPR-ALPR/UFPR-ALPR dataset/training/track0001/'
path = 'data/UFPR-ALPR/UFPR-ALPR dataset/training/'


for p in os.listdir(path):
        print(p)
        txt_paths = [path for path in os.listdir(os.path.join(path,p)) if ".txt" in path]

        for txt_path in txt_paths:
                print(path+p+'/'+txt_path)

# def get_plate_pos(txt_path):
#     txt_file = open(txt_path, "r")
#     txt_lines = txt_file.readlines()
#     for line in txt_lines:
#         if "position_plate" in line:
#             plate_pos = [int(s) for s in line.split() if s.isdigit()]
#             xmin,ymin,xmax,ymax = plate_pos[0], plate_pos[1], plate_pos[0]+plate_pos[2], plate_pos[1]+plate_pos[3]
#             return xmin,xmax,ymin,ymax

# txt_paths = [path for path in os.listdir(path) if ".txt" in path]
# for txt_path in txt_paths:
#         file_name, _ = os.path.splitext(txt_path)
#         img_path = file_name + '.png'
#         print(file_name,end=' ')
#         xmin,xmax,ymin,ymax = get_plate_pos(path+txt_path)
#         print(xmin,ymin,xmax,ymax)
#         image = cv.imread(path+img_path)
#         height,width = image.shape[:2]
#         print(height,width)
#         print(xmin/width,ymin/height,xmax/width,ymax/height)
#         cv.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),3)
#         plt.imshow(image)

#         xmin_vec, xmax_vec, ymin_vec, ymax_vec = [],[],[],[]
#         xmin_vec.append(xmin)
#         xmax_vec.append(xmax)
#         ymin_vec.append(ymin)
#         ymax_vec.append(ymax)
#         xmin_vec_norm = np.divide(xmin_vec, width) # List of normalized left x coordinates in bounding box (1 per box)
#         xmax_vec_norm = np.divide(xmax_vec, width) # List of normalized right x coordinates in bounding box # (1 per box)
#         ymin_vec_norm = np.divide(ymin_vec, height) # List of normalized top y coordinates in bounding box (1 per box)
#         ymax_vec_norm = np.divide(ymax_vec, height) # List of normalized bottom y coordinates in bounding box # (1 per box)
#         print(xmin_vec_norm,xmax_vec_norm,ymin_vec_norm,ymax_vec_norm)
# plt.show()