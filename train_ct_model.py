import numpy as np
import tensorflow as tf
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, Dense, Flatten, Reshape, UpSampling2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
from train_ct_generator import Generator
from train_ct_val_generator import Generator_val
#from sklearn.model_selection import train_test_split
# from keras.activations import linear

def loss_fuc():
    return

def unet():
    inputs = Input(batch_shape=(None, 512, 512, 1), name='inputs')
    # print(inputs.shape)
    conv1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    # print(conv1.shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)#32.512.512
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#32.256.256

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)#64.256.256
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#64.128.128

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)#128.128.128
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#128.64.64
    #
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)#256.64.64
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)#256.32.32
    #
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)#512.32.32
    drop5 = Dropout(0.5, name='drop5')(conv5)
    print(drop5.shape)
    
    #target two
    ######################################################################
    fullconv = Flatten()(drop5)
    fullconv = Dense(64, activation='relu')(fullconv)
    fullconv = Dropout(0.5)(fullconv)
    fctocov = Reshape((8, 8, 1))(fullconv)#1.8.8
    fctocov = UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(fctocov)#1.32.32
    conv_m = Conv2D(256, (3, 3), activation='relu', padding='same')(fctocov)#256.32.32
    ######################################################################
    deconv5_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv_m)#256.64.64
    up6_1 = concatenate([deconv5_1, drop4], axis=3)#512.64.64
    #
    conv6_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6_1)#256.64.64
    #
    #conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    up7_1 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv3], axis=3)#256.128.128
    conv7_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7_1)
    conv7_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7_1)#128.128.128
    drop7_1 = Dropout(0.5, name='drop7_1')(conv7_1)
    #
    #conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    up8_1 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(drop7_1), conv2], axis=3)#128.256.256
    conv8_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8_1)
    conv8_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8_1)#64.256.256
    #
    #conv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    up9_1 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8_1), conv1], axis=3)#64.512.512
    conv9_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9_1)
    conv9_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9_1)#32.512.512
    drop9_1 = Dropout(0.5)(conv9_1)

    ##target one
    #
    #up6 = Conv2D(256, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Upsampling2D(size = (2, 2))(drop5))
    #merge6 = concatenate([drop4, up6],)
    deconv5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5)#256,64,64
    up6 = concatenate([deconv5, drop4], axis=3)
    #
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)#256,64,64
    #
    #conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    
    #conv7_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop7_1)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), drop7_1], axis=3)#128,128,128->256,128,128
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)#128,128,128
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)#128,128,128
    drop7 = Dropout(0.5, name='drop7')(conv7)
    #
    #conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    #conv8_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8_1)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(drop7), conv8_1], axis=3)#64,256,256->128,256,256
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)#64,256,256
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)#64,256,256
    #
    #conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    #conv9_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9_1)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv9_1], axis=3)#32,512,512->64,512,512
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)#32,512,512
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)#32,512,512
    drop9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='outputs1')(drop9)
    conv10_1 = Conv2D(1, (1, 1), activation='relu', name='outputs2')(drop9_1)
    # x = Conv2D(256, (3, 3), padding='same', name='xxx1')(drop3)
    # x = Conv2D(128, (3, 3), padding='same', name='xxx2')(x)
    # x = MaxPooling2D((2, 2), name='xxx4')(x)

    # x = Flatten()(x)

    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1, activation='relu', name='outputs')(x)
    # x = Dense(1, activation='sigmoid', name='outputs2')(x)

    model = Model(inputs=[inputs], outputs=[conv10, conv10_1])
    return model

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score 

def dice_loss(y_true, y_pred):
    smooth = 1.
    e = 0.5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    loss1 = 1. - score
    loss2 = K.binary_crossentropy(y_true, y_pred)
    #loss2 = tf.math.log(dice_coef(y_true, y_pred))
    return e*loss1 + (1-e)*loss2

def tversky_loss(y_true, y_pred):
    alpha = 0.3
    beta  = 0.7
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32') 
    return Ncl-T

def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    model = unet()
    model.compile(optimizer=Adam(lr=1e-5),
                  loss={'outputs1': 'binary_crossentropy', 'outputs2': 'mse'},
                  metrics=['accuracy'])
    model.summary()
    #model = load_model('ct_unet.h5')
    x_path = './dataset/CT/images_previous1/'
    # x_path = './dataset/images/'
    y_path = './dataset/CT/masks/'
    y_generated_path = './dataset/CT/generated_images_previous1/'
    # param_path = './dataset/height_param/'
    x_val_path = './dataset/CT/val_images/'
    y_val_path = './dataset/CT/val_masks/'
    generator = Generator(x_path, y_path, y_generated_path)
    generator_val = Generator_val(x_val_path,y_val_path)
    model_checkpoint = ModelCheckpoint('models/ct_unet-{epoch:04d}.h5', monitor='loss', verbose=1, save_best_only=True, period = 1)
    # early_stopping = EarlyStopping('val_acc',min_delta=0.0001, patience=20, verbose=1, mode='max')
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001,verbose=1)
    #model.fit_generator(generator=generator.generator(), validation_data=generator_val.generator_val(), steps_per_epoch=500, epochs=200, validation_steps=10, verbose=1, callbacks=[model_checkpoint])
    model.fit_generator(generator=generator.generator(), steps_per_epoch=1000, epochs=500, verbose=1, callbacks=[model_checkpoint])
    # model.fit_generator(generator=generator.generator(), steps_per_epoch=4, epochs=500, verbose=1,
    #                     callbacks=[model_checkpoint])
