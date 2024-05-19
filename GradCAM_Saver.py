import pandas as pd
import numpy as np
import cv2                     
import os                  
from tqdm import tqdm  
import random
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Model, layers
from numpy import loadtxt
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from IPython.display import Image as imgdisp, display
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

## Size of training data of the models
size = (32,32)

# Define input layer
inputs = Input(shape=(32, 32, 3))

# Convolutional layers
conv1 = Conv2D(64, (3, 3), activation='relu')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
dropout1 = Dropout(0.25)(pool1)

conv2 = Conv2D(128, (3, 3), activation='relu')(dropout1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
dropout2 = Dropout(0.25)(pool2)

# Flatten layer
flatten = Flatten()(dropout2)

# Dense layers
dense1 = Dense(128, activation='relu')(flatten)
dropout3 = Dropout(0.5)(dense1)

# Output layer
outputs = Dense(7, activation='softmax')(dropout3)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

model.load_weights('model850(setup2_32_85).h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# <---------------------------------------------------------------------------------------------------------------------->
# <---------------------------------------------------------------------------------------------------------------------->

## Get image Numpy array :
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# <---------------------------------------------------------------------------------------------------------------------->
# <---------------------------------------------------------------------------------------------------------------------->

## Make the heatmap for an image array :
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs[0]], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)


    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# <---------------------------------------------------------------------------------------------------------------------->

## Second heatmap function, for code shrinking :
def generate_heatmap(img_path, last_conv_layer_name):
  img = get_img_array(img_path, size)

  # Remove last layer's softmax
  model.layers[-1].activation = None
  heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)

  return heatmap

# <---------------------------------------------------------------------------------------------------------------------->
# <---------------------------------------------------------------------------------------------------------------------->

## Main GRAD-MAP function (Image + heatmap layer save -> GRAD-CAM FOLDER (imageId_colormap.jpg) :
def save_gradcam(img_path, id, colormap, alpha, cam_path="./GRAD-CAM/"):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = generate_heatmap(img_path,"conv2d_1")
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use colorMap colormap to colorize heatmap
    colorMap = cm.get_cmap(f"{colormap}")

    # Use RGB values of the colormap
    colorMap_colors = colorMap(np.arange(256))[:, :3]
    colorMap_heatmap = colorMap_colors[heatmap]

    # Create an image with RGB colorized heatmap
    colorMap_heatmap = image.array_to_img(colorMap_heatmap)
    colorMap_heatmap = colorMap_heatmap.resize((img.shape[1], img.shape[0]))
    colorMap_heatmap = image.img_to_array(colorMap_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = colorMap_heatmap * alpha + img
    # superimposed_img = np.squeeze(superimposed_img, axis=0)
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img = superimposed_img.resize((img.shape[1], img.shape[0]))
    # Save the superimposed image
    superimposed_img.save(f"{cam_path}{str(id)}_{colormap}.jpg")
