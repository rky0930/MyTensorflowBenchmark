import os
from time import time
import numpy as np
import tensorflow as tf

class ObjectDetection(object):
    def __init__(self, config):
        # Model
        self.model_path = os.path.join(config['base_dir'], config['checkpoint'])
        self.confidence_score_threshold = config['confidence_score_threshold']    

        # Import Graph
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph)
        if os.path.isdir(self.model_path):
            print("use saved model", self.model_path)
            # use "saved model"
            with self.detection_graph.as_default():
                tf.saved_model.loader.load(
                    self.sess,
                    [tf.saved_model.tag_constants.SERVING],
                    self.model_path)
        else: # use "frozen graph
            print("use frozen")
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, "rb") as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name="")
                
        self.image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        self.boxes        = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        self.scores       = self.sess.graph.get_tensor_by_name('detection_scores:0')
        self.classes      = self.sess.graph.get_tensor_by_name('detection_classes:0')

    def preprocessing(self, inputs):
        x = np.expand_dims(inputs, axis=0)
        return x

    def run(self, frame):
        rgb_frame = frame[...,::-1] # BGR to RGB
        frame_np = np.array(rgb_frame, dtype=np.uint8)                   
        self.frame_height, self.frame_width, _ = frame.shape        
        preprocessed_frame = self.preprocessing(frame_np)
        start_t = time()
        (boxes, scores, classes) = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={self.image_tensor: preprocessed_frame})
        duration_t = time() - start_t  
        boxes, classes = self.postprocessing(boxes, scores, classes)
        return duration_t, boxes, classes
    
    def postprocessing(self, boxes, scores, classes):
        obj_indices = np.squeeze(np.argwhere(
        scores[0]>self.confidence_score_threshold), axis=1)
        boxes = boxes[0][obj_indices]
        classes = classes[0][obj_indices]
        print(classes)
        return boxes, classes

    def close(self):
        self.sess.close()
