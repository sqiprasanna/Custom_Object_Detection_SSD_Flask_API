import os
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import FPS 

ap = argparse.ArgumentParser()
args = {
       "config": './yolov3-spp.cfg',
       "weights": './yolov3-spp.weights',
       "classes": './coco.names'
       }
#args = ap.parse_args()


# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# Darw a rectangle surrounding the object and its class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load names classes
classes = None
with open(args['classes'], 'r') as f:
    classes = [line.strip() for line in f.readlines()]


#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args['weights'],args['config'])


def adjust_gamma(image, gamma=1):
   # build a lookup table mapping the pixel values [0, 255] to
   # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

count = 0
f_count = 0

cap = cv2.VideoCapture(0)
fps = FPS().start()
# files = os.listdir(path)
# files = files[2:]
# for i in files:
while True:
    # image = cv2.imread(path +'/'+ i)
    _,frame = cap.read()
    image = frame
    Width = 416
    Height = 416
    f_count += 1
    image = adjust_gamma(image)
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # display output image    
    cv2.imshow("object detection", image)

    # wait until any key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

fps.stop()
        
     # save output image to disk
    # cv2.imwrite("object-detection.jpg", image)

    # release resources
# print("Time Elapsed {:.2f}".format(fps.elapsed()))  
# print("Approx FPS {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()
    