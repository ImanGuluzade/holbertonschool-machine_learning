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

        # Get the input resolution of the network from the model object
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        img_h, img_w = image_size[0], image_size[1]

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            # Extract raw predictions from the tensor channels
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # 1. Calculate Sigmoid activation for center coordinates
            sig_x = 1 / (1 + np.exp(-t_x))
            sig_y = 1 / (1 + np.exp(-t_y))

            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_cls = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_conf)
            box_class_probs.append(box_cls)

            # 2. Build the mesh grid coordinate maps (cx, cy)
            cx = np.tile(np.arange(grid_w), (grid_h, 1))
            cx = cx.reshape(grid_h, grid_w, 1)

            cy = np.tile(np.arange(grid_h), (grid_w, 1)).T
            cy = cy.reshape(grid_h, grid_w, 1)

            # 3. Compute predicted center and dimensions relative to the grid
            bx = (sig_x + cx) / grid_w
            by = (sig_y + cy) / grid_h

            # Extract corresponding anchors for the current feature scale
            anchors_scale = self.anchors[i]
            pw = anchors_scale[:, 0]
            ph = anchors_scale[:, 1]

            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h

            # 4. Transform from center coordinates to (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            # Combine transformed elements back into box structure array
            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return (boxes, box_confidences, box_class_probs)
