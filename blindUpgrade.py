import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import re
from pygame import mixer
import time

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import ImageGrab



# # Model preparation 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
mixer.init()
old_name = ""
name = ""
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90
IMAGE_SIZE = (12, 8)

# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#intializing the web camera device

import cv2

#                                                                   #

cap = cv2.VideoCapture(1)

#     OpenCV VideoCapture                                           #


# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
      #scrgrb = ImageGrab.grab()
      #image_np =  np.array(ImageGrab.grab(bbox=(0, 40,600, 600)))
      ret,image_np = cap.read()
      
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      

      name = vis_util.get_name(image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      if not name == old_name:
              print(str(name))
              old_name = name
              try:
                      mixer.music.load("start" + ".mp3")  
                      mixer.music.play()
                      time.sleep(0.9)
                      for box in boxes:
                              xmax = round(float(box[0][3]), 3)
                              xmin = round(float(box[0][1]), 3)
                              if(round(xmax-xmin, 3) > 0.6):
                                      print("near")
                                      mixer.music.load("near" + ".mp3")  
                                      mixer.music.play()
                                      time.sleep(0.9)
                              elif(round(xmax-xmin, 3) < 0.3):
                                      print("far")
                                      mixer.music.load("far" + ".mp3")  
                                      mixer.music.play()
                                      time.sleep(0.9)
                              else:
                                      print("middle")
                                      mixer.music.load("middle" + ".mp3")  
                                      mixer.music.play()
                                      time.sleep(2)
                              #print(xmax+xmin/2)
                              if(xmax+xmin/2 >=1):
                                      print("right")
                                      mixer.music.load("right" + ".mp3")  
                                      mixer.music.play()
                                      time.sleep(0.9)
                              elif(xmax+xmin/2 <= 0.47):
                                      print("left")
                                      mixer.music.load("left" + ".mp3")  
                                      mixer.music.play()
                                      time.sleep(0.9)
                              else:
                                      print("center")
                                      mixer.music.load("center" + ".mp3")  
                                      mixer.music.play()
                                      time.sleep(0.9)
                              print(round(xmax-xmin, 3))
                              
                      mixer.music.load(name + ".mp3")  
                      mixer.music.play()  
                      time.sleep(1.5)
              except:
                      print("None sound. Add " + str(name) + ".mp3")
              
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      cv2.imshow('image', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break




