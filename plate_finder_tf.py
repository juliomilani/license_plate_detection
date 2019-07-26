import numpy as np
import tensorflow as tf
import cv2 as cv
import os

# Read the graph.
with tf.gfile.FastGFile('C:/tensorflow/plate_detector2/models/1306/out-7252/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.summary.FileWriter('logs', graph_def)

in_folder = 'C:/tensorflow/plate_detector2/teste_imgs/'
out_folder = 'C:/tensorflow/plate_detector2/teste_imgs_out/'
paths = [os.path.join(in_folder,image) for image in os.listdir(in_folder)]
paths_out = [os.path.join(out_folder,image) for image in os.listdir(in_folder)]


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
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.1:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                cv.putText(img,'{:03.2f}%'.format(score*100),(int(x),int(y)),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                print(score, bbox[1], bbox[0], bbox[3], bbox[2])

        cv.imshow('Image', img)
        cv.imwrite(paths_out[i],img)
        cv.waitKey()
        cv.destroyAllWindows()