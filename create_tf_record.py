import tensorflow as tf

from object_detection.utils import dataset_util

import glob
import os
import re
import cv2 as cv
import numpy as np
import io
import PIL.Image
import hashlib
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def get_plate_pos(txt_path):
    txt_file = open(txt_path, "r")
    txt_lines = txt_file.readlines()
    for line in txt_lines:
        if "position_plate" in line:
            plate_pos = [int(s) for s in line.split() if s.isdigit()]
            xmin,ymin,xmax,ymax = plate_pos[0], plate_pos[1], plate_pos[0]+plate_pos[2], plate_pos[1]+plate_pos[3]
            return xmin,xmax,ymin,ymax

def create_tf_example(img_path,xmin_vec,xmax_vec,ymin_vec,ymax_vec):
    print(img_path)
    img = cv.imread(img_path)
    img_filename = img_path.split(os.sep)[-1]
    img_name, img_extension = os.path.splitext(img_filename)

    with tf.gfile.GFile(img_path,'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = PIL.Image.open(encoded_png_io)
    key = hashlib.sha256(encoded_png).hexdigest()

    height,width = img.shape[:2]
    filename = img_filename
    encoded_image_data = encoded_png # Encoded image bytes 
    image_format = 'png'.encode('utf8')


    xmin_vec_norm = np.divide(xmin_vec, width) # List of normalized left x coordinates in bounding box (1 per box)
    xmax_vec_norm = np.divide(xmax_vec, width) # List of normalized right x coordinates in bounding box # (1 per box)
    ymin_vec_norm = np.divide(ymin_vec, height) # List of normalized top y coordinates in bounding box (1 per box)
    ymax_vec_norm = np.divide(ymax_vec, height) # List of normalized bottom y coordinates in bounding box # (1 per box)
    classes_text, classes = [],[]
    for _ in xmin_vec_norm:
        classes.append(1)             # List of integer class id of bounding box (1 per box)
        classes_text.append('placa'.encode('utf8'))  # List of string class name of bounding box (1 per box)

    if not np.all(np.array([len(classes),len(classes_text),len(xmin_vec_norm),len(xmax_vec_norm),len(ymin_vec_norm),len(ymax_vec_norm)]) == len(ymax_vec_norm)):
        print('Error: vector length mismatch')

    print(xmin_vec_norm,xmax_vec_norm,ymin_vec_norm,ymax_vec_norm,classes_text,classes)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_vec_norm),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_vec_norm),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_vec_norm),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_vec_norm),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    return tf_example


def main(_):
    training_path = 'data/UFPR-ALPR/UFPR-ALPR dataset/testing/'
        
    output_path = 'data/'
    output_filename = 'ufpr_testing.record'


    writer = tf.python_io.TFRecordWriter(output_path  + output_filename)

    

    for trck_i,track_folder in enumerate(os.listdir(training_path)):

        path = os.path.join(training_path,track_folder)+'/'
        paths_vec =  os.listdir(path)
        #list all paths with .txt
        txt_paths = [path for path in paths_vec if ".txt" in path] 

        for i,txt_path in enumerate(txt_paths):
            print('{}/{} @Track {}/{} :'.format(i+1,len(txt_paths),trck_i+1,len(os.listdir(training_path))),end=' ')
            file_name, _ = os.path.splitext(txt_path)
            img_path = path+file_name + '.png'
            txt_path = path+file_name + '.txt'
            image = cv.imread(img_path)
            xmin_vec, xmax_vec, ymin_vec, ymax_vec = [],[],[],[]
            
            xmin,xmax,ymin,ymax = get_plate_pos(txt_path)
            xmin_vec.append(xmin)
            xmax_vec.append(xmax)
            ymin_vec.append(ymin)
            ymax_vec.append(ymax)
                
            tf_example = create_tf_example(img_path,xmin_vec, xmax_vec, ymin_vec, ymax_vec)
            writer.write(tf_example.SerializeToString())
        
    writer.close()  


if __name__ == '__main__':
  tf.app.run()
