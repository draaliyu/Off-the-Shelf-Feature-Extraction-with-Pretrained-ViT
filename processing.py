# processing.py
import tensorflow as tf

BATCH_SIZE = 16


def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image


def preprocess(image, label, num_classes):
    image = tf.cast(image, tf.float32) / 255.0
    image = data_augment(image)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def create_datasets(X_train, X_val, y_train, y_val, num_classes):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_dataset = train_dataset.map(lambda x, y: preprocess(x, y, num_classes)).batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: preprocess(x, y, num_classes)).batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)

    return train_dataset, val_dataset
