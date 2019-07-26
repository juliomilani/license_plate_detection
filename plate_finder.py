 
import cv2
 
# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('C:/tensorflow/plate_detector2/models/1306/out/frozen_inference_graph.pb')
# , 'C:/tensorflow/plate_detector2/models/1306/train/graph.pbtxt')
 
# Input image
img = cv2.imread('C:/tensorflow/plate_detector2/data/UFPR-ALPR/UFPR-ALPR dataset/validation/track0061/track0061[01].png')
rows, cols, channels = img.shape
 
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
 
# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()
 
# Loop on the outputs
for detection in networkOutput[0,0]:
    
    score = float(detection[2])
    if score > 0.2:
    	
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
 
        #draw a red rectangle around detected objects
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
 
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()