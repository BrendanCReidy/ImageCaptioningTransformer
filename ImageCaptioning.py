import ImageLanguageTransformer

import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import re
import numpy as np
import time
import json
from glob import glob
from PIL import Image
import pickle
from PIL import Image
import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

eval_dir = "evaluation_images"
cocoPath = '/data'
annotation_folder = cocoPath + '/annotations'
image_folder = cocoPath + '/train2017'
annotation_file = annotation_folder + '/captions_train2017.json'

embedding_dim = 512
num_layers = 4
d_model = 256
dff = 1024
units = 1024
num_heads = 8
dropout_rate = 0.1
target_vocab_size = 10000
maximum_position_encoding = 128

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    #img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

top_k = 1000

tokenizer = None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

BATCH_SIZE = 64
BUFFER_SIZE = 1000

max_length = 40

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
                                    
transformer = ImageLanguageTransformer.Transformer(
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    embedding_dim=embedding_dim,
    target_vocab_size=target_vocab_size,
    maximum_position_encoding=maximum_position_encoding,
    units=units,
    num_layers=num_layers,
)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset

def add_noise(img, noise):
    img = img.numpy()
    img_arr = (img).astype(np.int8)
    im = Image.fromarray(img_arr, 'RGB')
    im.save("not_noisy.jpeg")

    w,h = img.shape[0], img.shape[1]
    total_size = w*h
    num_altered_pixels = int(noise*total_size)
    rands = random.sample(range(0, total_size), num_altered_pixels)
    xs = [x//w for x in rands]
    ys = [y%h for y in rands]
    for i in range(num_altered_pixels):
      x, y = xs[i], ys[i]
      img[x][y] = np.random.randint(low=0, high=255, size=(3,))

    img_arr = (img).astype(np.int8)
    im = Image.fromarray(img_arr, 'RGB')
    im.save("noisy.jpeg")
    return img


def predict(img, max_length=40, noise=0):
    temp_input = load_image(img)[0]
    temp_input = add_noise(temp_input, noise=noise)
    temp_input = tf.keras.applications.inception_v3.preprocess_input(temp_input)
    temp_input = tf.expand_dims(temp_input, 0)
    #print(temp_input)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    encoder_input = img_tensor_val

    tar = tf.convert_to_tensor(np.ones((1, 1))*tokenizer.word_index['<start>'])
    tar = tf.cast(tar, tf.float32)

    output = tar


    for i in range(max_length):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        mask = ImageLanguageTransformer.create_masks(tar)
        predictions = transformer(encoder_input, tar, mask, training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id = tf.cast(predicted_id, tf.float32)
        if(predicted_id.numpy()[0,:]==tokenizer.word_index['<end>']):
          break

        tar = tf.concat([tar, predicted_id], axis=-1)
    output = tar.numpy()[0,1:]
    #print(output)
    word = ""
    for value in output:
      word += tokenizer.index_word[int(value)] + " "
    return word

#for image_name in os.listdir(eval_dir):
#  predict(os.path.join(eval_dir, image_name))