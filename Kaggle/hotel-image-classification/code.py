#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:05:36 2016

@author: ashis
"""
# Please, read the README.txt file to set environment
from __future__ import print_function
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import SGD
from keras.constraints import maxnorm


from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import glob

import PIL
from PIL import Image

import csv
import re
#*************** Sorting file names *****************
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
# ************************************************************************

# Loading Training or Testing image/data
# parameter(image directory, flag as data type indicator)
def loadData(directory, flag):
    index = 0

    for filename in sorted(glob.glob(directory), key=numericalSort):       
        
        if index == loop_condition and flag ==1:
            break
        if index == test_sample_numbers and flag ==2:
            break
            
        # exception is used to avoid corrupted/damaged image file                  
        try:
            img = Image.open(filename)
            img.load()
        except Exception as ex:
            continue
           
        # resizing all original image in img_width by img_height
        img = img.resize((img_width, img_height), PIL.Image.ANTIALIAS)
        image_array = np.asarray( img )
        
        check_shape =image_array.shape 
        
        # avoiding an image if it is not in RGB format
        if len( check_shape) ==3  and check_shape[2]==3:
             
            sample_x = np.transpose(image_array, (2, 0, 1) )
            
            if flag ==1:        # loading training data
                X_train[index]  = sample_x
            else:               # loading testing data
                testingImage[index]  = sample_x
                filenameList[index]= filename  # saving image file name                     
            
            index +=1 
    
#   **********************************************************************
# loading Testing Y values from csv file
def loadYtrainData():
    reader = csv.DictReader( open('train.csv','r'), delimiter=',')
    index = 0
    for value in reader:
        
        if index == loop_condition:
            break
        for col in range( 1, class_number+1): 
            column_str = 'col'+ str(col)
            Y_train[index][col-1] = value[column_str] # [col-1], because start from 0
            # taking 0 to 7 column values of all row
        
        index +=1
    
# **********************************************************************
    
# saving final result in file (.csv) format
def generateOutput(yValue):
    index =0
   
    writer=open('output.csv','w')
    writer.write("id,col1,col2,col3,col4,col5,col6,col7,col8\n")
   
    for outputValue in yValue:
        idName = str( filenameList[index] )
        imageID=idName.split("/",2)
        imageID2 = imageID[2].split(".", 1)
        writer.write(imageID2[0])
                     
        for tem in range(0, 8):
            writer.write(","+ str(outputValue[tem]) )
        writer.write("\n")
        index+=1        
    writer.close()   

# ************************************************************************
def callCNNmodel():
    model = Sequential()

    # adding different layers
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    #passing input values to hidden layer
    model.add(Activation('relu'))   
    
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #adding more layers
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    
    
    model.add(Flatten())    # it changes our data into 1D format
    model.add(Dense(512, W_constraint=maxnorm(3) ) )
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_number))  # output layer
    model.add(Activation('softmax'))
    
    # using SGD with learning rate 0.01
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    # some preprocessing and realtime data augmentation parameters
    datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False, 
        rotation_range=0, 
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically 
        horizontal_flip=True,  # randomly flip images
        vertical_flip= False)  # randomly flip images

    datagen.fit(X_train)

    # fit the model 
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size) ,
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=epoch,
                        validation_data=(X_test, Y_test), # Taking test Data
                        nb_val_samples=X_test.shape[0]
                        )   
                        
    #getting prediction for test images
    yValue = model.predict(  testingImage, batch_size=32, verbose=0)
    generateOutput(yValue)  # saving result in file format
    
    

#   **********************************************************************  

if __name__ == '__main__':
    
    img_width, img_height = 100, 100     # dimensions of images.
    train_sample_numbers = 38000		  # 38372
    # some files are corrupted. So, taking less sample numbers
	
    validation_sample_numbers = 7200	  # Take almost 20% of total train data as validation sample  
    loop_condition = 38000
    test_sample_numbers =  19634        # Total test image: 19648
    # some test files are also corrupted like training data
    # So, taking less sample numbers, 19634

    batch_size = 32
    class_number = 8     # the number of objects
    epoch = 20	         # iteration time
    
    
    # fix random seed for reproducibility
    seed = 5
    np.random.seed(seed)

    
    # setting training and testing directory
    training_directory = 'data/train/*.jpg'
    testing_directory = 'data/testing/*.jpg'
    
    X_train = np.empty([train_sample_numbers, 3, img_width, img_height])
    Y_train = np.empty([train_sample_numbers, class_number])
    
    loadData(training_directory, 1) # 1 is a flag value to load train data
    loadYtrainData()

    
    # taking some sample for validation test
    X_test = X_train[(train_sample_numbers- validation_sample_numbers) : train_sample_numbers, ]
    Y_test = Y_train[(train_sample_numbers- validation_sample_numbers) : train_sample_numbers, ]
    
    X_train = X_train[0:(train_sample_numbers- validation_sample_numbers), ]
    Y_train = Y_train[0:(train_sample_numbers- validation_sample_numbers), ]
    
    # testingImage for testing data
    testingImage = np.empty([test_sample_numbers, 3, img_width, img_height])
    filenameList = [None] * test_sample_numbers # declareing empty List
        
    loadData(testing_directory, 2) # 2 is a flag value to load test data
    
    print ("Data load completed. Please, wait")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    testingImage = testingImage.astype('float32')
    
    # taking normalization value
    X_train /= 255 
    X_test /= 255
    testingImage /= 255
    
    # calling Convolutional Neural Network Model
    callCNNmodel()
    
    print ("Code End") 
