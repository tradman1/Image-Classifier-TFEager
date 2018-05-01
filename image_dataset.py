import tensorflow as tf
from tensorflow.contrib.data import shuffle_and_repeat, map_and_batch

img_features = {
    'image/height': tf.FixedLenFeature([], tf.int64),
    'image/width': tf.FixedLenFeature([], tf.int64),
    'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
    'image/channels':  tf.FixedLenFeature([], tf.int64),
    'image/class/label': tf.FixedLenFeature([],tf.int64),
    'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
    'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
    'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
}


class ImageDataset:

    def __init__(self, file_pattern, num_epochs, batch_size, image_size, shuffle_buff_size=5000):
        files = tf.data.Dataset.list_files(file_pattern)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=4)

        dataset = dataset.apply(shuffle_and_repeat(shuffle_buff_size, num_epochs))
        dataset = dataset.apply(
            map_and_batch(lambda x: self._parse_fn(x, image_size), batch_size, num_parallel_batches=4))

        self.dataset = dataset

    def _parse_fn(self, example, image_size):
        parsed = tf.parse_single_example(example, img_features)
        image = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, image_size)

        #augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)

        return image, parsed["image/class/label"]-1

    def get_dataset(self):
        return self.dataset

