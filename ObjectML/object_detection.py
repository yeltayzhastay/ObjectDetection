import numpy as np
import cv2
from .lib.utils import *
import os
import time
import base64

class Detection(object):
    def __init__(self):
        model_cfg = 'ObjectML/models/yolov3-tiny.cfg'
        model_weights = 'ObjectML/models/yolov3-tiny.weights'
        video = 'ObjectML/videos/Traffic - 28291.mp4'
        coco = 'ObjectML/models/coco.names'
        src = 0

        self.frameWidth = 1280
        self.frameHeight = 720

        self.net = cv2.dnn.readNet(model_weights, model_cfg)
        self.classes = []
        with open(coco, "r") as f:
            self.classes = [line.strip() for line in f.readlines()] # we put the names in to an array

        layers_names = self.net.getLayerNames()
        self.output_layers = [layers_names[i[0] -1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size = (len(self.classes), 3))

    def cap_set(self, camera):
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)


    def get_frame(self):
        # Object detection 
        success, frame = self.cap.read()

        frame = cv2.resize(frame, (self.frameWidth, self.frameHeight), None)
        height, width, channels = frame.shape
        # Detect image
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (480, 480), (0,0,0), swapRB = True, crop = False)
        self.net.setInput(blob)
        start = time.time()
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y -h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = "{}: {:.2f}%".format(self.classes[class_ids[i]], confidences[i]*100)
                color = self.colors[i]
                cv2.rectangle(frame, (x,y), (x+w+20, y+h+20), color, 2)
                cv2.putText(frame, label, (x,y+10), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        # cv2.imshow("Image", frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break