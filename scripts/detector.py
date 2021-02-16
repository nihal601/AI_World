import cv2
import time
import tensorflow as tf
from core import utils
import json
import numpy as np
import datetime
import os
import base64
import requests
import multiprocessing as mp
from tensorflow.python.keras import backend as k



from flask import Flask, request, jsonify
flaskserver = Flask(__name__)

from args import imgsize, batchsize
from args import vehicle_weights,vehicle_names,fsm_weights,fsm_names,yolo_weights,yolo_names
#from args import truck_weights,truck_names

# Handle API Call
@flaskserver.route(rule='/detect', methods=['POST'])
def detect():
    
    global detection
    content = request
    batch = np.fromstring(content.data, np.float32).reshape((-1, imgsize, imgsize, 3))
    inferenceList = detection.inference(batch)
    return jsonify({"Response":str(inferenceList)})

@flaskserver.route(rule='/detect2', methods=['POST'])
def detect2():

    global detection2
    content = request
    batch = np.fromstring(content.data, np.float32).reshape((-1, imgsize, imgsize, 3))
    inferenceList = detection2.inference(batch)
    return jsonify({"Response":str(inferenceList)})

@flaskserver.route(rule='/detect3', methods=['POST'])
def detect3():
    
    global detection3
    content = request
    batch = np.fromstring(content.data, np.float32).reshape((-1, imgsize, imgsize, 3))
    inferenceList = detection3.inference(batch)
    return jsonify({"Response":str(inferenceList)})

class Detection():
    '''
    Detection : Class to hold all detection related 
        member methods and variables.
    '''
    def __init__(self, model_path = '',
                  names_path = '', gpu_memory= 0.2, 
                  image_size = 416):
    
        '''
        Constructor : Called when class object is created.
        
        '''
        self.names_file = names_path
        self.img_size = image_size
        self.max_batch_size = batchsize
        self.num_classes = len(utils.read_names(names_path))
        self.model_file = model_path

        self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(
            tf.get_default_graph(), self.model_file,
                                                ["Placeholder:0", 
                                                 "concat_9:0", 
                                                 "mul_6:0"])
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        _ = self.sess.run(self.output_tensors, feed_dict={
            self.input_tensor: self.createRandomSample()})

    def createRandomSample(self):
        '''
        createRandomSample : Method to create random sample to
            initialize tensorflow session.
        
        Returns
        -------
        randomSample : numpy.ndarray
            
        '''
        randomSample = None
        for i in range(0, self.max_batch_size):
            if i == 0:
                randomSample = np.expand_dims(np.random.random_sample((
                    self.img_size, self.img_size, 3)), axis=0)
                continue
            randomSample = np.concatenate((randomSample, np.expand_dims(
                np.random.random_sample((self.img_size, self.img_size, 3)), 
                axis=0)), axis=0)
        return randomSample
        
    def inference(self, batch):
        '''
        inference : run detection on yolov3
        
        Parameters
        ----------
        batch : numpy.ndarray
            numpy.ndarray of shape (batch, img_size, img_size, 3)
        Returns
        -------
        inferenceList : list
            list of object bounding boxes
            with scores and labels
        '''
        
        inferenceList = list()
        print(batch.shape)
        boxes, scores = self.sess.run(self.output_tensors, 
                            feed_dict={self.input_tensor: batch})
        for i in range(0, len(scores)):
            t_boxes = boxes[i]
            t_boxes = np.expand_dims(t_boxes, axis=0)
            t_scores = scores[i]
            t_scores = np.expand_dims(t_scores, axis=0)
            t_boxes, t_scores, t_labels = utils.cpu_nms(t_boxes, t_scores,
                self.num_classes, score_thresh=0.3, iou_thresh=0.4)
            inferenceList.append([t_labels, t_scores, t_boxes])
        return {'Detections':inferenceList}

detection = Detection(vehicle_weights,vehicle_names)
detection2 = Detection(fsm_weights, fsm_names)
detection3 = Detection(yolo_weights,yolo_names)
print('------------------------------')
print('Server Ready.')
print('------------------------------')

if __name__ == '__main__':

    flaskserver.run(host='127.0.0.1',
                    port=6969,
                    debug=True)