# Neural Style Transfer

This project implements **Neural Style Transfer (NST)** using TensorFlow 2.x. Neural Style Transfer is an optimization technique used to take two images—a *content image* and a *style reference image* (such as a artwork by a famous painter)—and blend them together so the output image looks like the content image, but "painted" in the style of the style reference image.

## Directory Structure

* `neural_style_transfer/`
    * `0-neural_style.py`: Contains the `NST` class initialization, input validation logic, and image preprocessing/scaling capabilities.
    * `README.md`: Project description, environment setup, and utilization guides.

## Requirements

* Python 3.x
* NumPy
* TensorFlow 2.x
* Matplotlib

## Class Implementation Details

### `NST` Class
Manages the style transfer configuration, feature layers extraction, and data validation.

* **Public Class Attributes:**
    * `style_layers`: List of target VGG19 convolution layers used to extract style data.
    * `content_layer`: Target VGG19 layer used to extract spatial layout content data.

* **Methods:**
    * `__init__(self, style_image, content_image, alpha=1e4, beta=1)`: Validates image structures and model weighting values, and stores the preprocessed inputs.
    * `scale_image(image)`: Static method that rescales input array values to `[0, 1]` and proportionally scales the largest dimension to exactly 512 pixels using bicubic interpolation.
