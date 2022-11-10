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
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"]="1"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


embedding_dim = 512
num_layers = 4
d_model = 256
dff = 1024
units = 1024
num_heads = 8
dropout_rate = 0.1
target_vocab_size = 10000
maximum_position_encoding = 128

cocoPath = 'data/'
annotation_folder = cocoPath + '/annotations'
image_folder = cocoPath + '/train2017'
annotation_file = annotation_folder + '/captions_train2017.json'

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
  caption = f'<start> {val["caption"]} <end>'
  image_path = image_folder + "/" + '%012d.jpg' % (val['image_id'])
  image_path_to_caption[image_path].append(caption)


image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

train_image_paths = image_paths
print(len(train_image_paths))

train_captions = []
img_name_vector = []

for image_path in train_image_paths:
  caption_list = image_path_to_caption[image_path]
  train_captions.extend(caption_list)
  img_name_vector.extend([image_path] * len(caption_list))

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)

image_db = {}

# UNCOMMENT FOR FIRST RUN (GEN_IMAGES)
"""
for img, path in image_dataset:
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode('utf-8')
    #print(path_of_feature)
    #image_db[path_of_feature] = bf.numpy()
    np.save(path_of_feature, bf.numpy())
#"""

def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# UNCOMMENT FOR FIRST RUN (GEN_TOKENIZER)
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=target_vocab_size,
                                                  oov_token='<unk>',
                                                  filters='!\'#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#"""
tokenizer = None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
  img_to_cap_vector[img].append(cap)

# Create training and validation sets using an 80-20 split randomly.
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

img_name_train = []
cap_train = []
for imgt in img_name_train_keys:
  capt_len = len(img_to_cap_vector[imgt])
  img_name_train.extend([imgt] * capt_len)
  cap_train.extend(img_to_cap_vector[imgt])

img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
  capv_len = len(img_to_cap_vector[imgv])
  img_name_val.extend([imgv] * capv_len)
  cap_val.extend(img_to_cap_vector[imgv])

BATCH_SIZE = 64
BUFFER_SIZE = 1000
num_steps = len(img_name_train) // BATCH_SIZE
features_shape = 2048
attention_features_shape = 64

def map_func(img_name, cap):
  #img_name = img_name.decode('utf-8')
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  #img_tensor = image_db[img_name]
  return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


transformer = ImageLanguageTransformer.Transformer(
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    embedding_dim=embedding_dim,
    target_vocab_size=target_vocab_size,
    maximum_position_encoding=maximum_position_encoding,
    units=units,
    num_layers=num_layers
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
loss_plot = []

@tf.function
def train_step(img_tensor, tar):
  loss = 0
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  combined_mask = ImageLanguageTransformer.create_masks(tar_inp)

  # initializing the hidden state for each batch
  # because the captions are not related from image to image

  with tf.GradientTape() as tape:
      predictions = transformer(img_tensor, tar_inp, combined_mask, training=True)
      loss = loss_function(tar_real, predictions)
    
  total_loss = (loss / int(target.shape[1]))

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  return loss, total_loss

EPOCHS = 20
start_epoch = 0


def evaluate(img, max_length=40):
    temp_input = tf.expand_dims(load_image(img)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    encoder_input = img_tensor_val

    tar = tf.convert_to_tensor(np.ones((1, 1))*tokenizer.word_index['<start>'])
    tar = tf.cast(tar, tf.float32)

    print(encoder_input.shape)

    output = tar


    for i in range(max_length):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        mask = ImageLanguageTransformer.create_masks(tar)
        predictions = transformer(encoder_input, tar, mask, training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id = tf.cast(predicted_id, tf.float32)

        tar = tf.concat([tar, predicted_id], axis=-1)
    output = tar.numpy()[0,:]
    print(output)
    word = ""
    for value in output:
      word += tokenizer.index_word[int(value)] + " "
    print(word)

evaluate("evaluation_images/test8.png")
evaluate("evaluation_images/salad.jpg")

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
            ckpt_manager.save()
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)


    #if epoch % 5 == 0:
    ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    with open('lossPlot.txt', 'w') as filehandle:
      for value in loss_plot:
          filehandle.write('%s\n' % value)

ckpt_manager.save()


evaluate("test8.png")