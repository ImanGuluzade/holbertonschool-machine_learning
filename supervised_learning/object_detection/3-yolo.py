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
            box_class_probs: list of numpy.ndarrays containing probabilities.

        Returns:
            A tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            classes = np.argmax(scores, axis=-1)
            max_scores = np.max(scores, axis=-1)

            filtering_mask = max_scores >= self.class_t

            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(classes[filtering_mask])
            box_scores.append(max_scores[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max Suppression to remove highly overlapping redundant boxes.

        Parameters:
            filtered_boxes: numpy.ndarray of shape (?, 4)
            box_classes: numpy.ndarray of shape (?,)
            box_scores: numpy.ndarray of shape (?,)

        Returns:
            A tuple of (box_predictions, predicted_box_classes,
                        predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Find unique classes present in the filtered detections
        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Mask tracking only data points matching the current class loop
            cls_mask = box_classes == cls
            
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            # Sort indices based on box score descending order
            sorted_indices = np.argsort(cls_scores)[::-1]

            while len(sorted_indices) > 0:
                # Select index pointing to the highest scoring box
                best_idx = sorted_indices[0]
                
                box_predictions.append(cls_boxes[best_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[best_idx])

                # Stop evaluation loop if this is the last remaining index
                if len(sorted_indices) == 1:
                    break

                # Extract coordinate tensors for intersection calculations
                b1 = cls_boxes[best_idx]
                other_boxes = cls_boxes[sorted_indices[1:]]

                # Compute overlapping box edges
                x1 = np.maximum(b1[0], other_boxes[:, 0])
                y1 = np.maximum(b1[1], other_boxes[:, 1])
                x2 = np.minimum(b1[2], other_boxes[:, 2])
                y2 = np.minimum(b1[3], other_boxes[:, 3])

                # Calculate intersection dimension regions safely above 0
                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                # Compute Area of Union (Area1 + Area2 - Intersection)
                area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area_others = ((other_boxes[:, 2] - other_boxes[:, 0]) *
                               (other_boxes[:, 3] - other_boxes[:, 1]))
                union = area_b1 + area_others - intersection

                # Vectorized Calculation of Intersection over Union (IoU)
                iou = intersection / union

                # Filter out boxes with IoU less than NMS threshold
                keep_mask = iou < self.nms_t
                # Update remaining sorted indices list using the boolean mask
                sorted_indices = sorted_indices[1:][keep_mask]

        # Convert accumulated lists to unified numpy array structures
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)
