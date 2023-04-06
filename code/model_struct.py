# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 01:07:32 2021

@author: onee
"""

from keras import regularizers

from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout

def MultiCls(x, n_class):
    inputs = Input(shape=x.shape)

    main_network = Conv2D(32, kernel_size=(3,3), padding="same")(inputs)
    main_network = Activation("relu")(main_network)

    main_network = Conv2D(32, kernel_size=(3,3), padding="same")(main_network)
    main_network = Activation("relu")(main_network)

    main_network = MaxPooling2D(pool_size=(2,2))(main_network)

    main_network = Conv2D(64, kernel_size=(3,3), padding="same")(main_network)
    main_network = Activation("relu")(main_network)

    main_network = Conv2D(64, kernel_size=(3,3), padding="same")(main_network)
    main_network = Activation("relu")(main_network)

    main_network = MaxPooling2D(pool_size=(2,2))(main_network)

    main_network = Flatten()(main_network)
    main_network = Dense(128)(main_network)
    main_network = Activation('relu')(main_network)

    main_network = Dense(n_class)(main_network)
    out = Activation('softmax')(main_network)
       
    model = Model(inputs=inputs, outputs=out)

    return model

def MultiCls_color(x, n_class):
    from keras.models import Sequential
    weight_decay = 0.0005
    
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=x.shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('softmax'))
    
    model.load_weights('../model/cifar100vgg.h5')
    
    model.pop()
    model.pop()
    
    model.add(Dense(n_class))
    model.add(Activation('softmax'))
    
    return model