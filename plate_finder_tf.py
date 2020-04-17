import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import argparse
import pytesseract
import string



def plate_ocr(img):
    # Returns a String containing de plate number
    new_width = 500
    scale_ratio = new_width / img.shape[1]
    new_height = int(img.shape[0] * scale_ratio)
    dim = (new_width, new_height)
    img_out = cv.resize(img.copy(), dim, interpolation = cv.INTER_AREA) #Rescale
    plate = pytesseract.image_to_string(img,config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789')
    plate = plate.strip(string.ascii_lowercase)
    plate = [char for char in plate if char.isalnum()]
    plate = ''.join(plate)
    return plate
    
parser = argparse.ArgumentParser()

parser.add_argument("--path_in", help="Input folder")
parser.add_argument("--path_out", help="Output folder")
args = parser.parse_args()

if args.path_in is None:
    in_folder = 'teste_imgs/'
else:
    in_folder = args.path_in

if args.path_out is None:
    out_folder = 'teste_imgs_out/'
else:
    out_folder = args.path_out

paths = [os.path.join(in_folder,image) for image in os.listdir(in_folder)]
paths_out = [os.path.join(out_folder,image) for image in os.listdir(in_folder)]



# Read the graph.
with tf.gfile.FastGFile('models/out-7252/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.summary.FileWriter('logs', graph_def)


with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    for i,path in enumerate(paths):
        # Read and preprocess an image.
        # img = cv.imread('C:/tensorflow/plate_detector2/data/UFPR-ALPR/UFPR-ALPR-dataset/validation/track0077/track0077[01].png')
        img = cv.imread(path)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for j in range(num_detections):
            classId = int(out[3][0][j])
            score = float(out[1][0][j])
            bbox = [float(v) for v in out[2][0][j]]
            if score > 0.1:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                plate_img = img_out[int(y):int(bottom),int(x):int(right)]
                plate_number = plate_ocr(plate_img)
                cv.putText(img,'{:03.2f}%'.format(plate_number),(int(x),int(y)),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                print("Found plate ",score*100,"%")

            cv.imwrite(paths_out[i],img)
            print(paths_out[i]," saved")