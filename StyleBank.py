## StyleBank.py ##


import numpy as np
import tensorflow as tf
from glob import glob
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from datetime import datetime
import os

# using tensorflow.python.keras as this solves the shape bug in Conv2DTranspose : https://github.com/keras-team/keras/issues/6777
from tensorflow.python import keras
from InstanceNormalization import InstanceNormalization
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.contrib.layers import instance_norm
from tensorflow.python.keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, InputLayer, Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.losses import mse, binary_crossentropy
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import backend as K

import tensorboard
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input


class StyleBank(object):
    def __init__(self):
        ######################### Tunable Parameters ################################
        # General
        self.img_shape = (None, 128, 128, 3)  # (None, 512, 512, 3) ## Image Shape
        self.n_styles = 4  # 50 ## Number of styles in the bank
        self.n_content = 1000 ## Number of content images
        self.N_steps = 300000 ## Total number of training steps
        self.T = 2 ## Number of consecutive steps for training styles before training the AutoEncoder
        self.print_iter = 100 ## Log output
        self.Batch_Size = 4 ## Batch size
        self.Use_Batch_Norm = True ## Use batch normalization instead of instance normalization

        # LR
        self.LR_Initial = 0.01 ## Initial ADAM learning rate
        self.LR_Current = self.LR_Initial ## For logging
        self.LR_Decay = 0.8 ## LR decay
        self.LR_Update_Every = self.N_steps / 10 ## LR decay period

        # Loss
        self.Optimizer = optimizers.Adam(lr=self.LR_Initial) ## Optimizer for both branches
        self.LossAlpha = 0.025  # Content weight
        self.LossBeta = 1.2  # Style weight
        self.LossGamma = 1.0  # Total Variation weight
        ######################### \Tunable Parameters ################################

        self.StyleNetLoss = {k: None for k in range(self.n_styles)}
        self.StyleNetContentLoss = {k: None for k in range(self.n_styles)}
        self.StyleNetStyleLoss = {k: None for k in range(self.n_styles)}

        # Data
        self.Content_DB = None
        self.Style_DB = None
        self.Content_DB_path = './DB/content/'
        self.Style_DB_path = './DB/style/'
        self.Content_DB_list = glob(self.Content_DB_path + '*')
        self.Style_DB_list = glob(self.Style_DB_path + '*')

        # VGG
        self.VGG16 = None

        # auto-encoder
        self.encoder = None
        self.decoder = None

        # style bank
        self.style_bank = {k: None for k in range(self.n_styles)}

        self.StyleNet = {k: None for k in range(self.n_styles)}
        self.AutoEncoderNet = None

        # inputs - content and one for style
        self.KinputContent = None
        self.KinputStyle = None
        self.tfStyleIndices = None

        self.TensorBoardStyleNet = {k: None for k in range(self.n_styles)}
        self.TensorBoardAutoEncoder = None

    def initialize_placeholders(self):
        # initialize the content and style image tensors
        self.KinputContent = Input(shape=self.img_shape[1:], name="InputContent")
        self.KinputDecoded = None

    def build_models(self):
        ###########
        # Encoder #
        ###########
        print("Building Encoder")
        input_layer = Input(shape=self.img_shape[1:])
        t_encoder = Conv2D(32, (9, 9), strides=(1, 1), padding='same', use_bias=False)(input_layer)
        if self.Use_Batch_Norm:
            t_encoder = BatchNormalization()(t_encoder)
        else:
            t_encoder = InstanceNormalization()(t_encoder)
        t_encoder = Activation('relu')(t_encoder)
        t_encoder = Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(t_encoder)
        if self.Use_Batch_Norm:
            t_encoder = BatchNormalization()(t_encoder)
        else:
            t_encoder = InstanceNormalization()(t_encoder)
        t_encoder = Activation('relu')(t_encoder)
        t_encoder = Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False)(t_encoder)
        if self.Use_Batch_Norm:
            t_encoder = BatchNormalization()(t_encoder)
        else:
            t_encoder = InstanceNormalization()(t_encoder)
        t_encoder = Activation('relu')(t_encoder)
        self.encoder = Model(input_layer, t_encoder, name='Encoder')
        print(self.encoder.summary())

        ###########
        # Decoder #
        ###########
        print("Building Decoder")
        input_layer = Input(shape=self.encoder.layers[-1].output_shape[1:])
        t_decoder = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(input_layer)
        if self.Use_Batch_Norm:
            t_decoder = BatchNormalization()(t_decoder)
        else:
            t_decoder = InstanceNormalization()(t_decoder)
        t_decoder = Activation('relu')(t_decoder)
        t_decoder = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(t_decoder)
        if self.Use_Batch_Norm:
            t_decoder = BatchNormalization()(t_decoder)
        else:
            t_decoder = InstanceNormalization()(t_decoder)
        t_decoder = Activation('relu')(t_decoder)
        t_decoder = Conv2DTranspose(3, (9, 9), strides=(1, 1), padding='same', use_bias=False)(t_decoder)
        self.decoder = Model(input_layer, t_decoder, name='Decoder')
        print(self.decoder.summary())


        #############
        # StyleBank #
        #############
        for i in self.style_bank:
            print("Building Style {}".format(i))
            bank_name = "StyleBank{}".format(i)
            stylenet_name = "StyleNet{}".format(i)
            input_layer = Input(shape=self.encoder.layers[-1].output_shape[1:])
            t_style = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(input_layer)
            if self.Use_Batch_Norm:
                t_style = BatchNormalization()(t_style)
            else:
                t_style = InstanceNormalization()(t_style)
            t_style = Activation('relu')(t_style)
            t_style = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(t_style)
            if self.Use_Batch_Norm:
                t_style = BatchNormalization()(t_style)
            else:
                t_style = InstanceNormalization()(t_style)
            t_style = Activation('relu')(t_style)
            t_style = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(t_style)
            if self.Use_Batch_Norm:
                t_style = BatchNormalization()(t_style)
            else:
                t_style = InstanceNormalization()(t_style)
            t_style = Activation('relu')(t_style)
            self.style_bank[i] = Model(input_layer, t_style, name=bank_name)

            #########################
            # StyleBank Full Models #
            #########################
            input_layer = self.encoder.layers[0].output  # layers.Input(batch_shape=self.encoder.layers[0].input_shape)
            prev_layer = input_layer
            for layer in self.encoder.layers[1:]:
                prev_layer = layer(prev_layer)
            for layer in self.style_bank[i].layers[1:]:
                prev_layer = layer(prev_layer)
            for layer in self.decoder.layers[1:]:
                prev_layer = layer(prev_layer)
            self.StyleNet[i] = Model([input_layer], [prev_layer], name=stylenet_name)
            print(self.StyleNet[i].summary())

        ##########################
        # AutoEncoder Full Model #
        ##########################
        print("Building AutoEncoder")
        input_layer = self.encoder.layers[0].output  # layers.Input(batch_shape=self.encoder.layers[0].input_shape)
        prev_layer = input_layer
        for layer in self.encoder.layers[1:]:
            prev_layer = layer(prev_layer)
        for layer in self.decoder.layers[1:]:
            prev_layer = layer(prev_layer)
        self.AutoEncoderNet = Model([input_layer], [prev_layer], name='AutoEncoder')
        print(self.AutoEncoderNet.summary())

        ### VGG
        print("Importing VGG")
        self.VGG16 = VGG16(include_top=False, weights='imagenet', input_shape=self.img_shape[1:])

        print("Plotting Models")
        plot_model(self.AutoEncoderNet, to_file='Model_AutoEncoderNet.png', show_shapes=True)
        plot_model(self.VGG16, to_file='Model_VGG16.png', show_shapes=True)
        for i in self.style_bank:
            stylenet_model_file = "Model_StyleNet{}.png".format(i)
            plot_model(self.StyleNet[i], to_file=stylenet_model_file, show_shapes=True)

    def compile_models(self):
        print("Compiling models")

        def total_variation_loss(x):
            img_nrows = self.img_shape[1]
            img_ncols = self.img_shape[2]
            assert K.ndim(x) == 4
            if K.image_data_format() == 'channels_first':
                a = K.square(
                    x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
                b = K.square(
                    x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
            else:
                a = K.square(
                    x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
                b = K.square(
                    x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])

            return K.sum(K.pow(a + b, 1.25))

        def gram_matrix(x):
            assert K.ndim(x) == 4
            grams = list()
            for i in range(self.Batch_Size):
                img = x[i, :, :, :]
                if K.image_data_format() == 'channels_first':
                    features = K.batch_flatten(img)
                else:
                    features = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
                grams.append(K.dot(features, K.transpose(features)))
            gram = tf.keras.backend.stack(grams)
            return gram

        def stylenet_loss_wrapper(input_tensor):
            def stylenet_loss(S, O):
                style_loss = K.variable(0.0)
                content_loss = K.variable(0.0)
                vgg16_layers = [l for l in self.VGG16.layers]
                vgg16_layers = vgg16_layers[1:]
                FlI = input_tensor #self.encoder.layers[0].output
                FlS = S
                FlO = O
                for i in range(len(vgg16_layers)):
                    FlI = vgg16_layers[i](FlI)
                    FlS = vgg16_layers[i](FlS)
                    FlO = vgg16_layers[i](FlO)
                    if vgg16_layers[i] == self.VGG16.get_layer('block4_conv2') or self.VGG16.get_layer(
                            'block3_conv2') or self.VGG16.get_layer('block2_conv2') or self.VGG16.get_layer('block1_conv2'):
                        gram_mse = K.mean(K.square(gram_matrix(FlO) - gram_matrix(FlS)))
                        layer_channels = vgg16_layers[i].output_shape[3]
                        layer_size = vgg16_layers[i].output_shape[1] * vgg16_layers[i].output_shape[2]
                        gram_mse_norm = gram_mse / (2.0**2 * (layer_channels ** 2) * (layer_size ** 2))
                        style_loss = style_loss + gram_mse_norm

                    if vgg16_layers[i] == self.VGG16.get_layer('block4_conv2'):
                        content_loss = K.mean(K.square(FlO - FlI))
                        break
                tv_loss = total_variation_loss(O)
                return self.LossAlpha * content_loss + self.LossBeta * style_loss + self.LossGamma * tv_loss
            return stylenet_loss

        # Compile Models
        for i in self.style_bank:
            print("Compiling StyleBank {}".format(i))
            self.StyleNet[i].compile(optimizer=self.Optimizer, loss=stylenet_loss_wrapper(self.StyleNet[i].layers[0].output))
        self.AutoEncoderNet.compile(optimizer=self.Optimizer, loss=mse)
        print("Initial learning rates: StyleNet={}, AutoEncoder={}".format(K.eval(self.StyleNet[0].optimizer.lr),
                                                                           K.eval(self.AutoEncoderNet.optimizer.lr)))
    def get_batch_ids(self, batch_size, data_size):
        return np.random.choice(np.arange(0, data_size), size=batch_size, replace=False)

    def train_models(self):
        style_id = 0
        new_lr = self.LR_Initial
        for step in range(self.N_steps):
            style_ids = [style_id for i in range(self.Batch_Size)]
            batch_ids = self.get_batch_ids(self.Batch_Size, self.n_content)
            # Load the DB
            print("Loading DB, step {}...".format(step), end='')
            self.Content_DB = np.array([
                resize(
                    imread(self.Content_DB_list[batch_id]), self.img_shape[1:]
                ) for batch_id in batch_ids
            ])
	    
            style_im = resize( imread(self.Style_DB_list[style_id]), self.img_shape[1:] )
            self.Style_DB = np.array([
                style_im for style_id in style_ids
            ])

            print("Finished Loading DB")
            if step % (self.T + 1) != self.T:  # Train Style
                loss_style = self.StyleNet[style_id].train_on_batch(self.Content_DB, self.Style_DB)
                self.TensorBoardStyleNet[style_id].on_epoch_end(step, self.named_logs(self.StyleNet[style_id], loss_style))
            else:  # Train AE
                loss_autoencoder = self.AutoEncoderNet.train_on_batch(self.Content_DB, self.Content_DB)
                self.TensorBoardAutoEncoder.on_epoch_end(step, self.named_logs(self.AutoEncoderNet, loss_autoencoder))
                style_id += 1
                style_id = style_id % self.n_styles
            if step % self.print_iter == 0 and step != 0:
                print("step {0}, loss_style={1}, loss_autoencoder={2}, timestamp={3}".format(step, loss_style,
                                                                                              loss_autoencoder,
                                                                                              datetime.now()))
            if step % self.LR_Update_Every == 0 and step != 0:
                new_lr = new_lr * self.LR_Decay
                self.LR_Current = new_lr
                for i in self.style_bank:
                    K.set_value(self.StyleNet[i].optimizer.lr, new_lr)
                K.set_value(self.AutoEncoderNet.optimizer.lr, new_lr)
                print("Updating LR to: StyleNet={}, AutoEncoder={}".format(K.eval(self.StyleNet[0].optimizer.lr),
                                                                           K.eval(self.AutoEncoderNet.optimizer.lr)))
        for i in self.style_bank:
            self.TensorBoardStyleNet[i].on_train_end(None)
        self.TensorBoardAutoEncoder.on_train_end(None)

    def prepare_tensorboard(self):
        for i in self.style_bank:
            self.TensorBoardStyleNet[i] = keras.callbacks.TensorBoard(
                log_dir="tb_logs/stylenet_{}".format(i),
                histogram_freq=0,
                batch_size=self.Batch_Size,
                write_graph=True,
                write_grads=True
            )
            self.TensorBoardStyleNet[i].set_model(self.StyleNet[i])
        self.TensorBoardAutoEncoder = keras.callbacks.TensorBoard(
            log_dir="tb_logs/autoencoder",
            histogram_freq=0,
            batch_size=self.Batch_Size,
            write_graph=True,
            write_grads=True
        )
        self.TensorBoardAutoEncoder.set_model(self.AutoEncoderNet)

    def named_logs(self, model, logs):
        result = {}
        for l in zip(model.metrics_names, [logs]):
            result[l[0]] = l[1]
        return result

    def save_models(self):
        # serialize model to JSON
        if self.Use_Batch_Norm:
            out_mod_dir = 'Save_BatchNorm'
        else:
            out_mod_dir = 'Save_InstNorm'
        os.makedirs(out_mod_dir)
        ae_json = self.AutoEncoderNet.to_json()
        with open(os.path.join(out_mod_dir, "autoencoder.json"), "w") as json_file:
            json_file.write(ae_json)
        # serialize weights to HDF5
        self.AutoEncoderNet.save_weights(os.path.join(out_mod_dir, "autoencoder.h5"))
        for i in self.style_bank:
            ae_json = self.StyleNet[i].to_json()
            with open(os.path.join(out_mod_dir, "stylenet_{}.json".format(i)), "w") as json_file:
                json_file.write(ae_json)
            # serialize weights to HDF5
            self.StyleNet[i].save_weights(os.path.join(out_mod_dir, "stylenet_{}.h5".format(i)))
        print("Saved model to disk")

    def load_models(self):
        # load json and create model
        if self.Use_Batch_Norm:
            out_mod_dir = 'Save_BatchNorm'
        else:
            out_mod_dir = 'Save_InstNorm'
        ae_json = open(os.path.join(out_mod_dir,'autoencoder.json'), 'r')
        ae_model_json = ae_json.read()
        ae_json.close()
        ae_model = models.model_from_json(ae_model_json,
                                          custom_objects={'InstanceNormalization': InstanceNormalization})
        # load weights into new model
        ae_model.load_weights(os.path.join(out_mod_dir,"autoencoder.h5"))
        self.AutoEncoderNet = ae_model
        for i in self.style_bank:
            ae_json = open(os.path.join(out_mod_dir,"stylenet_{}.json".format(i)), 'r')
            ae_model_json = ae_json.read()
            ae_json.close()
            ae_model = models.model_from_json(ae_model_json,
                                              custom_objects={'InstanceNormalization': InstanceNormalization})
            # load weights into new model
            ae_model.load_weights(os.path.join(out_mod_dir,"stylenet_{}.h5".format(i)))
            self.StyleNet[i] = ae_model
        print("Loaded models from disk")
