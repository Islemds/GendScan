import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import shutil
from shutil import copyfile
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,ConfusionMatrixDisplay, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG19

EPOCHS = 100
IMG_SIZE = 224
BATCH_SIZE = 8


METRICS = [
    tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
]



def build_model_EfficientNetB0(num_classes, metrics=METRICS, output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Inputs
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Base model
    base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    base_model.trainable = False

    # Add new top layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", bias_initializer=output_bias, name="pred")(x)

    # Create and compile the model
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer, loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy", metrics=metrics)

    return model

def build_model_EfficientNetB4(num_classes, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Inputs and augmentation
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Base model
    base_model = EfficientNetB4(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    base_model.trainable = False

    # Rebuild top layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation="softmax",
                                    bias_initializer=output_bias,
                                    name="pred")(x)

    # Create and compile the model
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB4")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=optimizer, 
        loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy", 
        metrics=metrics
    )

    return model

def build_model_EfficientNetV2B3(num_classes, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Inputs and augmentation
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Base model
    base_model = EfficientNetV2B3(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    base_model.trainable = False

    # Rebuild top layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation="softmax",
                                    bias_initializer=output_bias,
                                    name="pred")(x)

    # Create and compile the model
    model = tf.keras.Model(inputs, outputs, name="EfficientNetV2B3")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=optimizer, 
        loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy", 
        metrics=metrics
    )

    return model

def build_model_EfficientNetV2L(num_classes, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Inputs and augmentation
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # aug_inputs = image_augmentation(inputs)

    # Base model
    model = EfficientNetV2L(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation="softmax" if num_classes > 2 else "sigmoid",
                                    bias_initializer=output_bias,
                                    name="pred")(x)

    # Create and compile the model
    model = tf.keras.Model(inputs, outputs, name="EfficientNetV2L")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=optimizer, 
        loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy", 
        metrics=metrics
    )

    return model

def build_model_InceptionV3(num_classes, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Inputs and augmentation
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Base model
    base_model = InceptionV3(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    base_model.trainable = False

    # Rebuild top layers
    x = tf.keras.applications.inception_v3.preprocess_input(base_model.output)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation="softmax",
                                    bias_initializer=output_bias,
                                    name="pred")(x)

    # Create and compile the model
    model = tf.keras.Model(inputs, outputs, name="InceptionV3")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=optimizer, 
        loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy", 
        metrics=metrics
    )

    return model