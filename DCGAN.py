#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
import time
from IPython import display
import tensorflow as tf
from keras.initializers import RandomNormal
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import (
    Reshape,
    Dense,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    BatchNormalization,
    MaxPooling2D,
    Reshape,
    Input,
    Dropout,
)
from keras.layers.advanced_activations import ReLU, LeakyReLU
from keras.layers import Input
from keras.models import Model


img_size = 64
noise_dim = 100


batch_size = 64
steps_per_epoch = 3000
epochs = 10000


DIR = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\GANs\PokemonDataset\pokemon_jpg\pokemon_jpg"
save_path = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\GANs\SimpleGANs\DCGANS_Pokemon"

optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)


def load_data(data_path):

    image_list = []
    file_list = os.listdir(data_path)
    for image in file_list:
        path = os.path.join(data_path, image)
        image_single = cv2.imread(path)
        image_single = cv2.resize(image_single, (img_size, img_size))
        image_list.append(image_single)

    return image_list


images = load_data(DIR)


def visualize_images(images):
    for image in images:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        break


visualize_images(images)


def create_train_data(images):
    x_train = np.array(images)
    x_train = x_train.astype("float32")
    x_train /= 127.5
    x_train -= 1
    img_rows, img_cols, channels = x_train[1].shape
    return x_train, img_rows, img_cols, channels


x_train, img_rows, img_cols, channels = create_train_data(images)


def generator_model():
    n = 8

    gen_model = Sequential()
    gen_model.add(Dense(n * n * 256, use_bias=False, input_shape=(noise_dim,)))
    gen_model.add(Reshape((n, n, 256)))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU())

    gen_model.add(
        Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False)
    )
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU())

    gen_model.add(
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)
    )
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU())
    gen_model.add(
        Conv2DTranspose(
            channels,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
        )
    )

    return gen_model


def discriminator_model():

    disc_model = Sequential()
    disc_model.add(
        Conv2D(64, (5, 5), padding="same", input_shape=(img_rows, img_cols, channels))
    )
    disc_model.add(LeakyReLU())
    disc_model.add(Dropout(0.3))

    disc_model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    disc_model.add(Flatten())
    disc_model.add(Dense(1, activation="sigmoid"))

    return disc_model


generator = generator_model()


generator.summary()


discriminator = discriminator_model()


discriminator.summary()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )


num_examples_to_generate = 16

seed = np.random.normal(0, 1, size=(num_examples_to_generate, noise_dim))


def train(dataset, epochs, save_after):

    generate_and_save_images(generator, 0, seed)

    for epoch in range(epochs):
        for image_batch in dataset:
            image_batch = tf.expand_dims(image_batch, axis=0)
            train_step(image_batch)

        if (epoch + 1) % save_after == 0:

            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, seed)

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        if predictions.shape[-1] == 3:
            plt.imshow(predictions[i] * 0.5 + 0.5)
        else:
            plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


train(x_train, epochs, save_after=100)


for x in x_train:
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    break


discriminator.save(
    r"C:\Users\Desktop\Desktop\JuPyter Notebooks\GANs\trained_checkpoint\disc_model"
)


generator.save(
    r"C:\Users\Desktop\Desktop\JuPyter Notebooks\GANs\trained_checkpoint\gen_model"
)


import imageio
import glob


gif_file = "dcgan.gif"

with imageio.get_writer(gif_file, mode="I") as writer:
    filenames = glob.glob("image*.png")
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


import tensorflow_docs.vis.embed as embed

embed.embed_file(gif_file)
