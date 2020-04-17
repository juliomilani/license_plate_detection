import streamlit as st
import numpy as np
import tensorflow as tf
import cv2 as cv
import pytesseract
import string


class Params:
    def __init__(self):
        self.disp_width = 500


# Read the graph.
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def read_graph():
    with tf.gfile.FastGFile('models/out-7252/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.summary.FileWriter('logs', graph_def)
    return graph_def

params = Params()
graph_def = read_graph()

def main():
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg",'jpeg','png','bmp','gif'])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_in = cv.imdecode(file_bytes, 1)
        img_out = find_plate(img_in)
        st.image(img_out, channels="BGR",width=params.disp_width)

def find_plate(img_in):    
    img_out = img_in.copy()
    rows = img_in.shape[0]
    cols = img_in.shape[1]
    inp = cv.resize(img_in, (512, 512))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
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
                "Plate found! Confidence",score,
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                plate_img = img_out[int(y):int(bottom),int(x):int(right)]
                plate_number = plate_ocr(plate_img)
                st.write(plate_number)
                st.image(plate_img, channels="BGR",width=params.disp_width)
                cv.putText(img_out,'{:03.2f}%'.format(score*100),(int(x),int(y)),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
                cv.rectangle(img_out, (int(x), int(y)), (int(right), int(bottom)), (0, 255, 0), thickness=4)
            else:
                "No plate found"         
    return img_out 

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

if __name__ == '__main__':
    main()
