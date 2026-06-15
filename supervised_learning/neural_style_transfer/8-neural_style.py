@staticmethod
    def scale_image(image):
        """
        Rescales an image to have pixel values between 0 and 1
        and centralizes its dimensions for VGG19 input
        """
        if (not isinstance(image, tf.Tensor) and
                not isinstance(image, tf.Variable)):
            if len(image.shape) != 3 or style_image.shape[2] != 3:
                raise TypeError(
                    "image must be a numpy.ndarray with shape (h, w, 3)"
                )
        elif len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        # 1. Cast to float32 first to guarantee precise mathematical interpolation
        image = tf.cast(image, tf.float32)
        
        # 2. Perform bicubic resizing
        image = tf.image.resize(image, [h_new, w_new], method='bicubic')
        
        # 3. Add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        # 4. Normalize pixels to [0, 1] range
        image = image / 255.0
        
        # 5. Clip to prevent floating point overshoot past limits
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
