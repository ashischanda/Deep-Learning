from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


from batches5 import load_batch_cate1  		# loading our file
from batches5 import load_batch_cate2  		# loading our file
from batches5 import load_batch_cate3     	# loading our file
from batches5 import load_cross_validation  	# loading our file


# ******************************************************************

IMG_WIDTH = 180
IMG_HEIGHT = 180
CHANNELS = 3        #RGB image

# We need to change here 
EPOCH_NUM = 10
CLASS_NUMBER = 5270
total_train_batch_file = 50
total_test_batch_file = 5
MODEL_NAME= "CNN_E10_B50_cate3.tfl"



'''
Note: The category ID is not 1 to 5270. Hence,  we need to convert these id into a sequence number
Y label is just a number, not a matrix. Hence, we used: Y = to_categorical(Y, CLASS_NUMBER)
'''

# ****************************************************************************

def build_network():   
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    
    # Convolutional network building
    network = input_data(shape=[None, IMG_WIDTH, IMG_HEIGHT, CHANNELS],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network,  CLASS_NUMBER , activation='softmax')
    
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    
    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model   
# ****************************************************************************
# ****************************************************************************
  

if __name__== "__main__":
 
  model = build_network()
  
  after_first_run_flag= False
  val_x, cat1_y, cate2_y, cate3_y = load_cross_validation()
  
  print ("Program started")
  # ***************************************************************************
  # (initial, stop, increment)
  for epoch in range(0, EPOCH_NUM ):
    for batch_index in range(0, total_train_batch_file):
        if (after_first_run_flag):
            model.load( MODEL_NAME )
            
        after_first_run_flag = True    
        val_set = None

        train_data, train_labels = load_batch_cate3(batch_index, True)  # We need to change here 
        
        #converting Y list into Matrix
        train_labels = to_categorical( train_labels, CLASS_NUMBER)
        val_y_labels = to_categorical( cate3_y, CLASS_NUMBER)           # We need to change here
        if (batch_index == total_train_batch_file-1 ):
          val_set =(val_x, val_y_labels)

        
        model.fit( train_data, train_labels, n_epoch= 1 , shuffle=True, validation_set= val_set ,
                show_metric=True, batch_size=96, run_id=MODEL_NAME )   # We need to change here 
        
        
        model.save( MODEL_NAME )
        print ("Finished load: ", batch_index )
    print ("Finished epoch: ", epoch )
  # ***************************************************************************    
  print ("training finished")
  
