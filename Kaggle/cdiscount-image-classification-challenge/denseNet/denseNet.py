from __future__ import division, print_function, absolute_import

import tflearn
#from tflearn.layers.conv import conv_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


import tensorflow as tf

def densenet_block(incoming, nb_layers, growth, bottleneck=True,
                   downsample=True, downsample_strides=2, activation='relu',
                   batch_norm=True, dropout=False, dropout_keep_prob=0.5,
                   weights_init='variance_scaling', regularizer='L2',
                   weight_decay=0.0001, bias=True, bias_init='zeros',
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="DenseNetBlock"):
    densenet = incoming

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        for i in range(nb_layers):

            # Identity
            conn = densenet

            # 1x1 Conv layer of the bottleneck block
            if bottleneck:
                if batch_norm:
                    densenet = tflearn.batch_normalization(densenet)
                densenet = tflearn.activation(densenet, activation)
                densenet = conv_2d(densenet, nb_filter=growth,
                                   filter_size=1,
                                   bias=bias,
                                   weights_init=weights_init,
                                   bias_init=bias_init,
                                   regularizer=regularizer,
                                   weight_decay=weight_decay,
                                   trainable=trainable,
                                   restore=restore)

            # 3x3 Conv layer
            if batch_norm:
                densenet = tflearn.batch_normalization(densenet)
            densenet = tflearn.activation(densenet, activation)
            densenet = conv_2d(densenet, nb_filter=growth,
                               filter_size=3,
                               bias=bias,
                               weights_init=weights_init,
                               bias_init=bias_init,
                               regularizer=regularizer,
                               weight_decay=weight_decay,
                               trainable=trainable,
                               restore=restore)

            # Connections
            densenet = tf.concat([densenet, conn], 3)

        # 1x1 Transition Conv
        if batch_norm:
            densenet = tflearn.batch_normalization(densenet)
        densenet = tflearn.activation(densenet, activation)
        densenet = conv_2d(densenet, nb_filter=growth,
                           filter_size=1,
                           bias=bias,
                           weights_init=weights_init,
                           bias_init=bias_init,
                           regularizer=regularizer,
                           weight_decay=weight_decay,
                           trainable=trainable,
                           restore=restore)
        if dropout:
            densenet = tflearn.dropout(densenet, keep_prob=dropout_keep_prob)

        # Downsampling
        if downsample:
            densenet = tflearn.avg_pool_2d(densenet, kernel_size=2,
                                           strides=downsample_strides)

    return densenet



# ****************************************************************************

from batches5 import load_batch_cate1  		# loading our file
from batches5 import load_batch_cate2  		# loading our file
from batches5 import load_batch_cate3     	# loading our file
from batches5 import load_cross_validation  	# loading our file


#                   load_batch is used for loading category 1

from tflearn.data_utils import to_categorical

# ******************************************************************

IMG_WIDTH = 180
IMG_HEIGHT = 180
CHANNELS = 3        #RGB image

# We need to change here 
EPOCH_NUM = 10
CLASS_NUMBER = 5270
total_train_batch_file = 50
total_test_batch_file = 1
MODEL_NAME= "DENSENET_E10_B50_class3.tfl"

k = 12		# Growth Rate (12, 16, 32, ...)
L = 10  	# Depth (40, 100, ...)
		# I used only 10 for Six layers
nb_layers = int((L - 4) / 3)

def build_network():   
        
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)
    
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([IMG_WIDTH, IMG_HEIGHT], padding=4)
    
    #Ashis: transition layer didn't use here
    #Hence, each densenet_block used same nb_layers and growth (k)
    #transition layer needs to balance two adjacent densenet_block

    #by default, dropout is set as false. Downsample is used as True

    # Building Residual Network
    net = input_data(shape=[None, IMG_WIDTH, IMG_HEIGHT, CHANNELS], data_preprocessing=img_prep, data_augmentation=img_aug)
    net = conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = densenet_block(net, nb_layers, k)
    #no transition layer 
    net = densenet_block(net, nb_layers, k)
    #no transition layer 

    #net = densenet_block(net, nb_layers, k)   #Ignore one

    #no transition layer 
    net = tflearn.global_avg_pool(net)
    
    # Regression
    net = fully_connected(net, CLASS_NUMBER , activation='softmax')
    #opt = tflearn.optimizers.Nesterov(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    #opt = tflearn.optimizers.AdaGrad (learning_rate=0.01, initial_accumulator_value=0.01)
    
    net = regression(net, optimizer="adam", loss='categorical_crossentropy', learning_rate=0.001)
    # Training
    model = tflearn.DNN(net, checkpoint_path='model_densenet',  max_checkpoints=10, tensorboard_verbose=0, clip_gradients=0.)
    return model



if __name__== "__main__":
 
  model = build_network()
  
  after_first_run_flag= True
  val_x, cate1_y, cate2_y, cate3_y = load_cross_validation()
  
  print ("Program started")
  # ***************************************************************************
  # (initial, stop, increment)
  for epoch in range(0, EPOCH_NUM ):
    for batch_index in range(0, total_train_batch_file):
        if (after_first_run_flag):
            model.load( MODEL_NAME )
            
        after_first_run_flag = True    
        train_data, train_labels = load_batch_cate3(batch_index, True)  # We need to change here 

        train_labels = to_categorical( train_labels, CLASS_NUMBER)
        val_y_labels= to_categorical( cate3_y, CLASS_NUMBER)           # We need to change here 
        
        model.fit(train_data, train_labels, n_epoch=1 , validation_set= (val_x, val_y_labels),
              show_metric=True, batch_size= 96, shuffle=True,
              run_id=MODEL_NAME)
   
        
        model.save( MODEL_NAME )
        print ("Finished load: ", batch_index )
    print ("Finished epoch: ", epoch )
	# ***************************************************************************    
  print ("training finished")
    
    
