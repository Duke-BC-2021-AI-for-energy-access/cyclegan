import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source')
parser.add_argument('--target')
parser.add_argument('--source_num_imgs', default=100)
parser.add_argument('--target_num_imgs', default=75)

opt = parser.parse_args()

'''
A training script that encompasses what is done in cyclegan.ipynb as of Nov 21

data.zip should be downloaded and extracted before running this script, using the method mentioned in the readme

Finally, a conda environment should be created for this script based on cyclegan.yml
'''
import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

import numpy as np
from PIL import Image
import glob

############### USER PARAMETERS ###############
BATCH_SIZE = 1
IMG_WIDTH = 608
IMG_HEIGHT = 608
SOURCE_DATASET = opt.source
TARGET_DATASET = opt.target
SOURCE_NUM_IMAGES = int(opt.source_num_imgs)
TARGET_NUM_IMAGES = int(opt.target_num_imgs)

print(f'Train: src: {SOURCE_DATASET}, targ: {TARGET_DATASET}, src_n: {SOURCE_NUM_IMAGES}, targ_n: {TARGET_NUM_IMAGES}')
###############################################


## /jitter/wt/images/EM/Real/*.jpg
## /jitter/wt/images/NE/Background/*.jpg

############# DATA FILE SPECIFICATION for original images ###################
# src_folders = {
#     'NE': 'Train NE Val NW 100 real 75 syn',
#     'NW': 'Train NW Val NE 100 real 75 syn',
#     'EM': 'Train EM Val EM 100 real 75 syn',
#     'SW': 'Train SW Val EM 100 real 75 syn'
#     }
# 
# # Globs the target images
# import glob
# 
# bg_folders = ['EM', 'NE', 'NW', 'SW']
# train_targ_filenames = glob.glob(f'colab-cyclegan-data/backgrounds/{TARGET_DATASET}/*')
# 
# ## Gets the train_src_filenames from training_img_paths.txt. This current implementation only takes the txt file in the NE domain. 
# txt_file = './colab-cyclegan-data/' + src_folders[SOURCE_DATASET] + '/baseline/training_img_paths.txt'
# 
# with open(txt_file) as f:
#   lines = f.readlines()
#   lines = [l.strip() for l in lines]
#   lines = ['colab-cyclegan-data' + l[2:] for l in lines]
# 
# train_src_filenames = lines
# del lines
#############################################################

############# DATA FILE SPECIFICATION for jitter images ###################

import json
json_file = '/work/yl708/bass/cyclegan/colab-cyclegan-data/jitter/domain_overview.json'

with open(json_file, 'r') as f:
    j = json.load(f)

src_numbers = j[SOURCE_DATASET]['Real']
target_numbers = j[TARGET_DATASET]['Background']

train_src_filenames = [f'/work/yl708/bass/cyclegan/colab-cyclegan-data/jitter/images/{SOURCE_DATASET}/Real/{s}.jpg' for s in src_numbers]
train_targ_filenames = [f'/work/yl708/bass/cyclegan/colab-cyclegan-data/jitter/images/{TARGET_DATASET}/Background/{t}.jpg' for t in target_numbers]

# train_src_filenames = glob.glob(f'/work/yl708/bass/cyclegan/colab-cyclegan-data/jitter/images/{SOURCE_DATASET}/Real/*.jpg')
# train_targ_filenames = glob.glob(f'/work/yl708/bass/cyclegan/colab-cyclegan-data/jitter/images/{TARGET_DATASET}/Background/*.jpg')

### USE ALL OF THE TRAIN
train_src_filenames = train_src_filenames[:SOURCE_NUM_IMAGES]

train_targ_filenames = train_targ_filenames[:TARGET_NUM_IMAGES]

#############################################################

import random
random.shuffle(train_src_filenames)

# selects number of training images equal to the number of images in train_targ, and use the rest as test data
# test_src_filenames = train_src_filenames[len(train_targ_filenames):]
# train_src_filenames = train_src_filenames[:len(train_targ_filenames)]
test_src_filenames = train_src_filenames[-30:] ## Choosing the last 30 train souce images as test images
train_src_filenames = train_src_filenames[:-30]

train_source = tf.data.Dataset.from_tensor_slices((train_src_filenames)) #75 train src images (from total 100)
train_target = tf.data.Dataset.from_tensor_slices((train_targ_filenames)) # 75 total target background images

print(test_src_filenames)
test_source = tf.data.Dataset.from_tensor_slices((test_src_filenames)) # 25 "target" src images (the remaining train ones)
# test_target = tf.data.Dataset.from_tensor_slices((test_targ_filenames)) # dont need it

def read_pixel_data(filepath):
  # Convert the compressed string to a 3D uint8 tensor
  # print(filepath.eval())

  img = tf.io.read_file(filepath)
  img = tf.io.decode_png(img, channels=3)
  img.set_shape([608, 608, 3])
  img = tf.image.resize(img, [608, 608])
  #img = tf.image.resize(img, [512, 512])
  return img
  # return filepath

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[608, 608, 3])
      #image, size=[512, 512, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  print(image)
  # resizing to 286 x 286 x 3

  # Originally 560 x 560
  image = tf.image.resize(image, [665, 665],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 512 x 512 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(imagepath):
  image = read_pixel_data(imagepath)
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(imagepath):
  image = read_pixel_data(imagepath)
  image = normalize(image)
  return image

train_source = train_source.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

train_target = train_target.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

test_source = test_source.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

# test_target = test_target.map(
    # preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

sample_source = next(iter(train_source))
sample_target = next(iter(train_target))

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_target = generator_g(sample_source)
to_source = generator_f(sample_target)

contrast = 8

imgs = [sample_source, to_target, sample_target, to_source]
title = ['Source', 'To Target', 'Target', 'To Source']

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = f"./s_{SOURCE_DATASET}_t_{TARGET_DATASET}_sn_{SOURCE_NUM_IMAGES}_tn_{TARGET_NUM_IMAGES}_checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


EPOCHS = 200

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_source, train_target)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))
