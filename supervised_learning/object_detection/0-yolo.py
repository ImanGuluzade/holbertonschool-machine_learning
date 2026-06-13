#!/usr/bin/env python3
"""
Defines the Yolo class to perform object detection using YOLO v3.
"""
import tensorflow.keras as K


class Yolo:
    """
    Yolo class constructor and properties for version 3 object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo model parameters and paths.

        Parameters:
            model_path: path to where a Darknet Keras model is stored.
            classes_path: path to where the list of class names is located.
            class_t: float representing the box score threshold.
            nms_t: float representing the IOU threshold for NMS.
            anchors: numpy.ndarray containing all anchor box dimensions.
        """
        # Load the pre-trained Darknet Keras model without compilation config
        self.model = K.models.load_model(model_path, compile=False)

        # Parse the classes file into a list of strings
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Set public instance thresholds and configurations
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
