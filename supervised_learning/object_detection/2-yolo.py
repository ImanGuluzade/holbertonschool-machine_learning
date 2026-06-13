#!/usr/bin/env python3
"""
Defines the Yolo class to perform object detection using YOLO v3.
"""
import numpy as np
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

    def process_outputs(self, outputs, image_size):
        """
        Processes the raw Darknet model outputs.

        Parameters:
            outputs: list of numpy.ndarrays containing model predictions.
            image_size: numpy.ndarray containing the image's original size
                        [image_height, image_width].

        Returns:
            A tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        img_h, img_w = image_size[0], image_size[1]

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            sig_x = 1 / (1 + np.exp(-t_x))
            sig_y = 1 / (1 + np.exp(-t_y))

            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_cls = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_conf)
            box_class_probs.append(box_cls)

            cx = np.tile(np.arange(grid_w), (grid_h, 1))
            cx = cx.reshape(grid_h, grid_w, 1)

            cy = np.tile(np.arange(grid_h), (grid_w, 1)).T
            cy = cy.reshape(grid_h, grid_w, 1)

            bx = (sig_x + cx) / grid_w
            by = (sig_y + cy) / grid_h

            anchors_scale = self.anchors[i]
            pw = anchors_scale[:, 0]
            ph = anchors_scale[:, 1]

            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h

            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes down by thresholding their class scores.

        Parameters:
            boxes: list of numpy.ndarrays containing boundary boxes.
            box_confidences: list of numpy.ndarrays containing box confidences.
            box_class_probs: list of numpy.ndarrays containing class probabilities.

        Returns:
            A tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Box scores = box confidence * class probabilities
            scores = box_confidences[i] * box_class_probs[i]

            # Track the index of the highest scoring class for each anchor box
            classes = np.argmax(scores, axis=-1)
            # Extract the actual value of those maximum scores
            max_scores = np.max(scores, axis=-1)

            # Create a boolean masking array where scores exceed the threshold
            filtering_mask = max_scores >= self.class_t

            # Apply the mask and flatten multi-dimensional structures down to (?,)
            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(classes[filtering_mask])
            box_scores.append(max_scores[filtering_mask])

        # Concatenate array elements from all feature map scales into unified outputs
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)
