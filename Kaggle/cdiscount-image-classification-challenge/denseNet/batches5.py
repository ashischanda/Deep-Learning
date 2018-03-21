import pickle
import numpy as np
import io

# Simple data processing
PATH = "scratch/data/"
training_directory = 'train.bson' # for training example use 'train_example.bson'
testing_directory = 'test.bson'


def load_batch_cate1(batch_no, flag):
    Y = []
    if flag:
        filename = training_directory
    else:
        filename = testing_directory
    X = pickle.load(open(PATH + 'X' + (str)(batch_no) + filename, 'rb'))
    Y = pickle.load(open(PATH + "cate1_" +'Y' + (str)(batch_no) + filename, 'rb') )
    return X, Y

def load_batch_cate2(batch_no, flag):
    Y = []
    if flag:
        filename = training_directory
    else:
        filename = testing_directory
    X = pickle.load(open(PATH + 'X' + (str)(batch_no) + filename, 'rb'))
    Y = pickle.load(open(PATH + "cate2_" +'Y' + (str)(batch_no) + filename, 'rb') )
    return X, Y


def load_batch_cate3(batch_no, flag):
    Y = []
    if flag:
        filename = training_directory
    else:
        filename = testing_directory
    X = pickle.load(open(PATH + 'X' + (str)(batch_no) + filename, 'rb'))
    Y = pickle.load(open(PATH +'Y' + (str)(batch_no) + filename, 'rb') )   #cate3 is in Y0train file
    return X, Y

def load_cross_validation():
    X = pickle.load(open('scratch/data/XcrossValidation' + training_directory, 'rb'))
    cate1_Y = pickle.load(open('scratch/data/cate1_YcrossValidation' + training_directory, 'rb'))
    cate2_Y = pickle.load(open('scratch/data/cate2_YcrossValidation' + training_directory, 'rb'))
    cate3_Y = pickle.load(open('scratch/data/cate3_YcrossValidation' + training_directory, 'rb'))
    
    return X, cate1_Y, cate2_Y, cate3_Y

#X, cate1_Y, cate2_Y, cate3_Y = load_cross_validation()
#print ("len ", len( cate1_Y ))



