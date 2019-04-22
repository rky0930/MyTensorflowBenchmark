import os
from time import time
import numpy as np
import tensorflow as tf

class ObjectDetection(object):
    def __init__(self, config):
        # Model
        self.model_path = os.path.join(config['base_dir'], config['checkpoint'])
        self.confidence_score_threshold = config['confidence_score_threshold']    

        self.set_graph()
        self.inference_counter = 0
        self.total_inference_t = 0

    def set_graph(self):
        # Import Graph
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph)
        if os.path.isdir(self.model_path):
            # use "saved model"
            with self.detection_graph.as_default():
                tf.saved_model.loader.load(
                    self.sess,
                    [tf.saved_model.tag_constants.SERVING],
                    self.model_path)
        else: # use "frozen graph
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, "rb") as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name="")
                
        self.image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        self.boxes        = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        self.scores       = self.sess.graph.get_tensor_by_name('detection_scores:0')
        self.label_ids    = self.sess.graph.get_tensor_by_name('detection_classes:0')

    def get_average_inference_time(self):
        if self.inference_counter == 0:
            return 0
        return round(self.total_inference_t / float(self.inference_counter), 3)

    def get_fps(self):
        avg_inference_t = self.get_average_inference_time()
        if avg_inference_t == 0:
            return 0
        return round(1 / avg_inference_t, 3)
    
    def reset_fps(self):
        self.inference_counter = 0
        self.total_inference_t = 0

    def preprocessing(self, inputs):
        x = np.expand_dims(inputs, axis=0)
        return x

    def run(self, image):
        rgb_image = image[...,::-1] # BGR to RGB
        image_np = np.array(rgb_image, dtype=np.uint8)                   
        self.image_height, self.image_width, _ = image.shape        
        preprocessed_image = self.preprocessing(image_np)
        start_t = time()
        (boxes, scores, label_ids) = self.sess.run(
            [self.boxes, self.scores, self.label_ids],
            feed_dict={self.image_tensor: preprocessed_image})
        self.total_inference_t += time() - start_t 
        self.inference_counter += 1 
        boxes, scores, label_ids = self.postprocessing(boxes, scores, label_ids)
        return boxes, scores, label_ids
    
    def postprocessing(self, boxes, scores, label_ids):
        obj_indices = np.squeeze(np.argwhere(
        scores[0]>self.confidence_score_threshold), axis=1)
        boxes = boxes[0][obj_indices]
        boxes[:,[0,2]] *= self.image_height # ymin, ymax
        boxes[:,[1,3]] *= self.image_width # xmin, xmax
        boxes = np.rint(boxes[:, [1,0,3,2]]).astype(np.int) # xmin, ymin, xmax, ymax
        
        scores = scores[0][obj_indices]
        label_ids = label_ids[0][obj_indices]
        return boxes, scores, label_ids

    def close(self):
        self.sess.close()
