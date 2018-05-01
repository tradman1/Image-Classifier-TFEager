import sys

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from image_dataset import ImageDataset
layers = tf.keras.layers

tf.enable_eager_execution()


def get_model(input_shape, n_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(6, (5, 5), strides=(1, 1), padding="valid", input_shape=input_shape),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding="valid"),
        layers.Activation("relu"),

        layers.Conv2D(16, (5, 5), strides=(1, 1), padding="valid"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding="valid"),
        layers.Activation("relu"),

        layers.Flatten(),
        layers.Dense(120),
        layers.Activation("relu"),
        layers.Dropout(0.4),
        layers.Dense(84),
        layers.Activation("relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes),
        layers.Activation("softmax")
    ])
    return model


def loss(model, x, y):
    pred = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=pred)


def train(model, dataset, optimizer):
    for (i, (x, y)) in enumerate(tfe.Iterator(dataset)):
        optimizer.minimize(lambda: loss(model, x, y),
                           global_step=tf.train.get_or_create_global_step())

        if i % 10 == 0:
            print("Loss at step {}: {:.5f}".format(i, loss(model, x, y)))

    print("Finished training! Final loss: {:.5f}".format(loss(model, x, y)))


def evaluate(model, dataset):
    acc = tfe.metrics.Accuracy()
    for x, y in tfe.Iterator(dataset):
        acc(tf.argmax(model(x), axis=1), y)

    print("Accuracy: {:.5f}".format(acc.result()))


def main(data_dir, n_classes=2, input_shape=(64, 64, 3), num_epochs=20, batch_size=64):
    model = get_model(input_shape, n_classes)
    img_dim = [input_shape[0], input_shape[1]]
    train_dataset = ImageDataset(data_dir + "/train*",
                                 num_epochs,
                                 batch_size,
                                 img_dim).get_dataset()
    optimizer = tf.train.AdamOptimizer()

    train(model, train_dataset, optimizer)

    test_dataset = ImageDataset(data_dir + "/validation*",
                                1,
                                batch_size,
                                img_dim).get_dataset()
    evaluate(model, test_dataset)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    main(data_dir)
