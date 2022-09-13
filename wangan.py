# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:18:12 2022

@author: katha
"""
# Copyright 2019 The TensorFlow Authors.
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import time
import tensorflow as tf
from IPython import display
import copy
import networkx as nx
import pickle
import random
import pandas as pd

# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
def reset_random_seeds(s):
   os.environ['PYTHONHASHSEED']=str(s)
   tf.random.set_seed(s)
   np.random.seed(s)
   random.seed(s)


def WANGAN(name, adj_matrix, batch_size = 200, learning_rate = 0.001, epochs = 1000, samples = 10000, RGB = True, s = 0):  
    reset_random_seeds(s)
    disc_losses = []
    gen_losses = []
    networks = []
    if RGB: colorchannels = 2
    else: colorchannels = 1
    nodes = len(adj_matrix)
    
    
    # pad networks/matrices/images so that they are divisible by 4
    nodes_rem = nodes % 4
    
    if nodes_rem == 0:
        matrix_size = nodes
        padding = 0
    else:
        padding = 4 - nodes_rem
        matrix_size = nodes + padding
    
    maximum_dist = np.max(adj_matrix)
    adj_matrix = (adj_matrix / maximum_dist) # this is equal to min/max normalization, as the diagional axis will always be 0 = np.min
    network = nx.Graph(adj_matrix)
    
             
    for i in range(samples + 1):
        permutation = tuple(random.sample(sorted(network), len(network)))
        mapping = dict(zip(network, permutation))
        permuted = nx.relabel_nodes(network, mapping) 
        permuted = nx.to_numpy_matrix(permuted,nodelist=list(range(len(network))))
        permuted = np.pad(permuted, [(0, padding), (0, padding)], mode='constant')
        
        perm_unweighted = copy.deepcopy(permuted)
        perm_unweighted[perm_unweighted > 0]=1

        
        if RGB:
            rgb = np.stack((np.asarray((perm_unweighted - .5) *2),np.asarray((permuted - .5)*2)), axis=-1)
            networks.append(rgb)
        else:
            bw = np.asarray((perm_unweighted - .5) *2)
            bw = bw[:, :, np.newaxis]
            networks.append(bw)

    
    networks = np.asarray(networks)
    train_images = networks
    
    
    BUFFER_SIZE = 10000
    BATCH_SIZE = batch_size
    
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    k = matrix_size
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(int(k/4)*int(k/4)*128*colorchannels, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Reshape((int(k/4), int(k/4), 128*colorchannels)))
        assert model.output_shape == (None, int(k/4), int(k/4), 128*colorchannels)  # Note: None is the batch size
        
        if RGB:
            # from k/4 x k/4 x 256 to k/4 x k/4 x 128
            model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            assert model.output_shape == (None, int(k/4), int(k/4), 128)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
    
        # from k/4 x k/4 x 128 to k/2 x k/2 x 64
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(k/2), int(k/2), 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        # from k/2 x k/2 x 64 to k x k x c (1 or 2)
        model.add(layers.Conv2DTranspose(colorchannels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, int(k), int(k), colorchannels)
    
        return model
    
    generator = make_generator_model()
    
    
    def make_discriminator_model():
        model = tf.keras.Sequential()
        
        # from k x k x c (1 or 2) to k/2 x k/2 x 64
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[int(k), int(k), colorchannels]))
        assert model.output_shape == (None, int(k/2), int(k/2), 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
    
        # from k/2 x k/2 x 64 to k/4 x k/4 x 128
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, int(k/4), int(k/4), 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        if RGB:
            # from k/4 x k/4 x 128 to k/4 x k/4 x 256
            model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same'))
            assert model.output_shape == (None, int(k/4), int(k/4), 256)
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))
    
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
    
        return model
    
    discriminator = make_discriminator_model()

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    
    EPOCHS = epochs
    noise_dim = 100
    num_examples_to_generate = 100
    
    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim],seed=s)
    
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = generator(noise, training=True)

          real_output = discriminator(images, training=True)
          fake_output = discriminator(generated_images, training=True)
          
          gen_loss = generator_loss(fake_output)
          disc_loss = discriminator_loss(real_output, fake_output)
    
          gen_losses.append(gen_loss)
          disc_losses.append(disc_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
    def train(dataset, epochs):
      for epoch in range(epochs):
        start = time.time()
    
        for image_batch in dataset:
          train_step(image_batch)
    
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,epoch + 1,seed)
        
        # Save the model every 100 epochs
        if (epoch + 1) % 100 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
    
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
      # Generate after the final epoch
      display.clear_output(wait=True)
      generate_and_save_images(generator,epochs,seed)
      
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)
    
        if (epoch % 100 == 0):

            for i in range(predictions.shape[0]):
                if i < 16:
                    plt.subplot(4, 4, i+1)
                    plt.axis('off')
                    if RGB:
                        plt.imshow( np.dstack((np.asarray(predictions[i]* .5 + .5),np.asarray(np.zeros(shape=(int(k),int(k),1))))))
                    else:
                        plt.imshow(np.asarray(predictions[i]* .5 + .5), cmap='gray_r')

                if (epoch % epochs == 0):
                    if RGB:
                        with open(name+"\\sample_at_epoch_"+str(epoch)+"_"+str(i)+"_"+str(s)+"_RGB.pkl","wb") as f:
                            pickle.dump(np.dstack((np.asarray((predictions[i]* .5 + .5)),np.asarray(np.zeros(shape=(int(k),int(k),1))))),f)
                    else:
                        with open(name+"\\sample_at_epoch_"+str(epoch)+"_"+str(i)+"_"+str(s)+"_BW.pkl","wb") as f:
                            pickle.dump(np.asarray((predictions[i]* .5 + .5)),f)

                    
                    if RGB:
                        plt.savefig(name+'\\image_at_epoch_'+str(epoch)+'_'+str(s)+'_RGB.pdf'.format(epoch),bbox_inches='tight',pad_inches=0.0)
                    else:
                        plt.savefig(name+'\\image_at_epoch_'+str(epoch)+'_'+str(s)+'_BW.pdf'.format(epoch),bbox_inches='tight',pad_inches=0.0)
            plt.show()
    
    train(train_dataset, EPOCHS)