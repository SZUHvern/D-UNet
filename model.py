from __future__ import print_function
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
import math
import h5py
from keras import backend as K
import tensorflow as tf
import math
from matplotlib import pyplot as plt
import time
import random
import copy
import os

def expand(x):
    x = K.expand_dims(x, axis=-1)
    return x


def squeeze(x):
    x = K.squeeze(x, axis=-1)
    return x


def BN_block(filter_num, input):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def BN_block3d(filter_num, input):
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def D_Add(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Add()([x, input2d])
    return x


def D_concat(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, input2d])
    x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def D_SE_concat(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = squeeze_excite_block(x)
    input2d = squeeze_excite_block(input2d)
    x = Concatenate()([x, input2d])
    x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def D_Add_SE(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Add()([x, input2d])
    x = squeeze_excite_block(x)
    return x


def D_SE_Add(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = squeeze_excite_block(x)
    input2d = squeeze_excite_block(input2d)
    x = Add()([x, input2d])

    return x


def D_concat_SE(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, input2d])
    x = squeeze_excite_block(x)
    x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def D_Unet():
    inputs = Input(shape=(192, 192, 4))
    input3d = Lambda(expand)(inputs)
    conv3d1 = BN_block3d(32, input3d)

    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

    conv3d2 = BN_block3d(64, pool3d1)

    pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

    conv3d3 = BN_block3d(128, pool3d2)


    conv1 = BN_block(32, inputs)
    #conv1 = D_Add(32, conv3d1, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(64, pool1)
    conv2 = D_SE_Add(64, conv3d2, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(128, pool2)
    conv3 = D_SE_Add(128, conv3d3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(256, pool3)
    conv4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BN_block(512, pool4)
    conv5 = Dropout(0.3)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate()([conv4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # conv10作为输出
    model = Model(input=inputs, output=conv10)

    return model


def Unet():
    inputs = Input(shape=(192, 192, 1))
    conv1 = BN_block(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(64, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(128, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(256, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BN_block(512, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # conv10作为输出

    model = Model(input=inputs, output=conv10)

    return model


def origin_block(filter_num, input):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x1 = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = Activation('relu')(x)
    return x


def Unet_origin():
    inputs = Input(shape=(192, 192, 1))
    conv1 = origin_block(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = origin_block(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = origin_block(256, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = origin_block(512, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = origin_block(1024, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = origin_block(512, merge6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = origin_block(256, merge7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = origin_block(128, merge8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = origin_block(64, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # conv10作为输出

    model = Model(input=inputs, output=conv10)

    return model

def Unet3d():
    inputs = Input(shape=(192, 192, 4))
    input3d = Lambda(expand)(inputs)
    conv1 = BN_block3d(32, input3d)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = BN_block3d(64, pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = BN_block3d(128, pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

    conv4 = BN_block3d(256, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)

    conv5 = BN_block3d(512, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='6')(
        UpSampling3D(size=(2, 2, 1))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block3d(256, merge6)

    up7 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='8')(
        UpSampling3D(size=(2, 2, 1))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block3d(128, merge7)

    up8 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='10')(
        UpSampling3D(size=(2, 2, 1))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block3d(64, merge8)

    up9 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='12')(
        UpSampling3D(size=(2, 2, 1))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block3d(32, merge9)
    conv10 = Conv3D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Lambda(squeeze)(conv10)
    # '''
    # conv11 = Lambda(squeeze)(conv10)
    conv11 = BN_block(32, conv10)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv11)
    # '''
    model = Model(input=inputs, output=conv11)

    return model

def SegNet(nClasses=1, input_height=192, input_width=192):
    img_input = Input(shape=(input_height, input_width, 1))
    kernel_size = 3
    # encoder
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 128x128
    x = Conv2D(128, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 64x64
    x = Conv2D(256, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 32x32
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 16x16
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 8x8

    # decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = Conv2D(nClasses, (1, 1), padding='valid',
               kernel_initializer='he_normal')(x)

    x = Activation('sigmoid')(x)
    model = Model(img_input, x, name='SegNet')
    return model


def conv_bn_block(x, filter):
    x = Conv3D(filter, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, x1])
    return x

def fcn_8s(num_classes=1, vgg_weight_path=None):
    img_input = Input(shape=(192, 192, 1))

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1',kernel_initializer='he_normal')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv3',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_3_out = MaxPooling2D()(x)

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1',kernel_initializer='he_normal')(block_3_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv3',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_4_out = MaxPooling2D()(x)

    # Block 5
    x = Conv2D(256, (3, 3), padding='same', name='block5_conv1',kernel_initializer='he_normal')(block_4_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block5_conv2',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block5_conv3',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, x)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # Convolutinalized fully connected layer.
    x = Conv2D(512, (6, 6), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Classifying layers.
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    block_3_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',kernel_initializer='he_normal')(block_3_out)
    block_3_out = BatchNormalization()(block_3_out)

    block_4_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',kernel_initializer='he_normal')(block_4_out)
    block_4_out = BatchNormalization()(block_4_out)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_4_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_3_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 8, x.shape[2] * 8)))(x)

    x = Activation('sigmoid')(x)

    model = Model(img_input, x)

    return model
