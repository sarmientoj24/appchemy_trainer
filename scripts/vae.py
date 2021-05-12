from keras.layers import Lambda, Input, Dense, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Activation, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.models import load_model
from keras import layers, initializers
from keras.layers import Dropout
import numpy as np
from keras.models import Model
import os
from matplotlib import pyplot as plt

    """
    Variational Autoencoder Class
    """
class VAE():
    image_shape = (80, 60, 3)

    def __init__(self, load=False, encoder_h5=None, 
                decoder_h5=None):

        if load:
            print("Loading model from h5...")
            self.load_model(encoder_h5, layer_ = 'encoder')
            self.load_model(decoder_h5, layer_ = 'decoder')

        else:
            print("Creating model architecture...")

            # Create network architecture parameters
            kernel_size = 2
            latent_dim = 16
            filter_dims = [16, 32, 64]
            filter_decoder = [64, 32, 16]
            image_shape = (240, 180, 3)

            inputs = Input(shape=image_shape, name='encoder_input')
            x = inputs
            for i in range(2):
                filters = filter_dims[i]
                x = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU(0.1)(x)

            # shape info needed to build decoder model
            shape = K.int_shape(x)

            # generate latent vector Q(z|X)
            x = Flatten()(x)
            x = Dense(128)(x)
            x = LeakyReLU(0.1)(x)
            z_mean = Dense(latent_dim, name='z_mean')(x)
            z_log_var = Dense(latent_dim, name='z_log_var',
                kernel_initializer=initializers.Constant(0),
                bias_initializer=initializers.Constant(0))(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

            # instantiate encoder model
            self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
            # encoder.summary()
            plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

            # build decoder model
            latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
            x = Dense(128)(latent_inputs)
            x = LeakyReLU(0.1)(x)
            x = Dense(shape[1] * shape[2] * shape[3])(x)
            x = LeakyReLU(0.1)(x)
            x = Reshape((shape[1], shape[2], shape[3]))(x)

            for i in range(2):
                filters = filter_decoder[i]
                x = Conv2DTranspose(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU(0.1)(x)

            outputs = Conv2DTranspose(filters=3,
                                    kernel_size=kernel_size,
                                    activation='sigmoid',
                                    padding='same',
                                    name='decoder_output')(x)

            # instantiate decoder model
            self.decoder = Model(latent_inputs, outputs, name='decoder')
            # decoder.summary()
            plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

            # instantiate VAE model
            outputs = decoder(encoder(inputs)[2])
            self.vae = Model(inputs, outputs, name='vae')
            # vae.summary()

            reconstruction_loss = mse(K.flatten(inputs),
                                                    K.flatten(outputs))

            reconstruction_loss *= image_shape[0] * image_shape[1]
            kl_loss = 1 + (z_log_var) - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)

            # def vae_loss(y_true, y_pred):
            #     """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
            #     recon = K.sum(K.binary_crossentropy(y_true, y_pred))
            #     kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma, axis=1)
            #     return recon + kl

            self.vae.add_loss(vae_loss)
            self.vae.compile(optimizer='adam')

            def sampling(args):
                z_mean, z_log_var = args
                batch = K.shape(z_mean)[0]
                dim = K.int_shape(z_mean)[1]
                # by default, random_normal has mean=0 and std=1.0
                epsilon = K.random_normal(shape=(batch, dim))
                return z_mean + K.exp(z_log_var / 2) * epsilon

    def get_summary(layer='vae'):
        if layer == 'vae':
            self.vae.summary()
        elif layer == 'decoder':
            self.decoder.summary()
        else:
            self.encoder.summary()


    def load_model(self, h5_file, layer_):
        print("Loading h5 ", layer_)
        if layer_ == 'encoder':
            self.encoder = load_model(h5_file)
        else:
            self.decoder = load_model(h5_file)


    def _fit(self, epochs, batch_size, x_train):
        self.history = self.vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1)
        self.vae.save('./vae_mnist.h5')
        self.encoder.save('./vae_encoder.h5')
        self.decoder.save('./vae_decoder.h5')


    def get_history(self):
        return self.history


    def plot_loss_function(self):
        # Plot training & validation loss values
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()


    def get_encoder_model(self):
        return self.encoder


    def get_decoder_model(self):
        return self.decoder


    def get_vae_model(self):
        return self.vae