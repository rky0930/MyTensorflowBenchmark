import os
from time import time
import numpy as np
import cv2
import json
import tensorflow as tf
from utils.label_map_tools import load_label_map

class ObjectDetection(object):
    def __init__(self, config, save_inference_result=False):
        # Model
        self.model_path = os.path.join(config['base_dir'], config['checkpoint'])
        self.confidence_score_threshold = config['confidence_score_threshold']    
        self.tflite    = config['tflite']
        self.quantized = config['quantized']
        self. attribute, self.label_id_to_name = load_label_map(
            os.path.join(config['base_dir'], config['label_map_file']))
        self.dtype = np.uint8 if self.quantized else np.float32
        self.set_model_size()
        if self.tflite:
            self.set_tensorflow_lite_graph()
        else:
            self.set_tensorflow_graph()
        self.inference_counter = 0
        self.total_inference_t = 0
        self.save_inference_result = save_inference_result
        if self.save_inference_result:
            print("save_inference_result:", self.save_inference_result)
            self.inference_result = {}
            self.inference_result_file = config['inference_result_file']

    def set_model_size(self):
        if os.path.isdir(self.model_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.model_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            self.model_size = total_size
        else:
            self.model_size = os.path.getsize(self.model_path)

    def set_tensorflow_graph(self):
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
        self.image_tensor   = self.sess.graph.get_tensor_by_name('image_tensor:0')
        self.boxes          = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        self.scores         = self.sess.graph.get_tensor_by_name('detection_scores:0')
        self.label_ids      = self.sess.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.sess.graph.get_tensor_by_name('num_detections:0')

    def set_tensorflow_lite_graph(self):
        self.interpreter = tf.contrib.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.tflite_input_height = int(self.input_details[0]['shape'][1])
        self.tflite_input_width  = int(self.input_details[0]['shape'][2])
            
    def get_average_inference_time(self):
        if self.inference_counter == 0:
            return 0
        return round(self.total_inference_t / float(self.inference_counter), 3)
    
    def reset_fps(self):
        self.inference_counter = 0
        self.total_inference_t = 0

    def preprocessing(self, input_image):
        if self.tflite:
            input_image = cv2.resize(input_image, 
            (self.tflite_input_height, self.tflite_input_width), 
             interpolation=cv2.INTER_AREA)
            if not self.quantized:
                input_image =  (input_image - 128.) / 128. #(2.0 / 255.0) * input_image - 1.0 
        rgb_image = input_image[...,::-1] # BGR to RGB
        image_np = np.array(rgb_image, dtype=self.dtype)     
        x = np.expand_dims(image_np, axis=0)
        return x
    
    def sess_run(self, preprocessed_image):
        if self.tflite:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], preprocessed_image)
            self.interpreter.invoke()
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            label_ids = self.interpreter.get_tensor(self.output_details[1]['index']) + 1
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
            
        else:
            boxes, scores, label_ids, num_detections = self.sess.run(
            [self.boxes, self.scores, self.label_ids, self.num_detections],
            feed_dict={self.image_tensor: preprocessed_image})
            num_detections = int(num_detections)
        # print(num_detections)
        # print(scores)
        boxes = boxes[0][:num_detections]
        label_ids = label_ids[0][:num_detections]
        scores = scores[0][:num_detections]
        return boxes, scores, label_ids

    def run(self, image_path):
        # Get image
        image = cv2.imread(image_path)
        self.image_height, self.image_width, _ = image.shape
        preprocessed_image = self.preprocessing(image)
        start_t = time()
        (boxes, scores, label_ids) = self.sess_run(preprocessed_image)
        self.total_inference_t += time() - start_t 
        self.inference_counter += 1 
        boxes, scores, label_ids = self.postprocessing(boxes, scores, label_ids)
        if self.save_inference_result:
            self.add_annotation(image_path, boxes, label_ids)
        return boxes, scores, label_ids
    
    def postprocessing(self, boxes, scores, label_ids):
        obj_indices = np.squeeze(np.argwhere(
        scores>self.confidence_score_threshold), axis=1)
        boxes = boxes[obj_indices]
        boxes[:,[0,2]] *= self.image_height # ymin, ymax
        boxes[:,[1,3]] *= self.image_width # xmin, xmax
        boxes = np.rint(boxes[:, [1,0,3,2]]).astype(np.int) # xmin, ymin, xmax, ymax
        
        scores = scores[obj_indices]
        label_ids = label_ids[obj_indices]
        return boxes, scores, label_ids

    def close(self):
        self.sess.close()

    def add_annotation(self, image_path, boxes, label_ids):
        image_name = os.path.basename(image_path)
        image_size = os.path.getsize(image_path)
        regions = self.raw_to_regions(boxes, label_ids)
        anno_key = "{}{}".format(image_name, image_size)
        self.inference_result[anno_key] = {
            "filename": image_name,
            "size": image_size,
            "regions": regions,
            "file_attributes": {}
        }

    def raw_to_regions(self, boxes, label_ids):
        regions = {}
        for idx, (box, label_id) in enumerate(zip(boxes, label_ids)):
            try:
                xmin, ymin, xmax, ymax = box
                height = ymax - ymin
                width = xmax - xmin
                label_name = self.label_id_to_name[label_id.astype(int)]
                region = {
                    "shape_attributes": {
                        "name": "rect",
                        "x": int(round(xmin)),
                        "y": int(round(ymin)),
                        "width":  int(round(width)),
                        "height": int(round(height))
                    },
                    "region_attributes": {
                        self.attribute: {
                            "0" : label_name
                        }
                    }
                }
                regions[idx] = region
            except ValueError:
                continue 
            except KeyError as e:
                print("KeyError: {}".format(e))
                
        return regions    

    def write_to_file(self):
        with open(self.inference_result_file, "w") as ret_fid:
            json.dump(self.inference_result, ret_fid, ensure_ascii=False)
        print("Inference result file is created: {}".format(self.inference_result_file))