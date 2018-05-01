# Image-Classifier-TFEager
A simple image classifier implemented using Tensorflow's eager execution.

Data input pipeline was done using tf.data.Dataset API. Images were converted to TFRecord files using [this script](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py).

Links:
  * Eager execution guide: https://www.tensorflow.org/programmers_guide/eager#train_a_model
  * tf.data tutorial: https://www.youtube.com/watch?v=uIcqeP7MFH0&t=803s
