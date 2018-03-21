from __future__ import division, print_function, absolute_import

import tflearn
import batches as b
import pickle
from paramsNo import calculate_params_no
from layers.resnext_block import resnext_block

# Number of resNext blocks
n = 2
last_model_file = 'resNext707Batches/ALLmodel_resnext_epoch_1.txt'
test_batches_no = 177

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()

# Building Residual Network
net = tflearn.input_data(shape=[None, b.IMG_WIDTH, b.IMG_HEIGHT, b.CHANNELS], data_preprocessing=img_prep, data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)

#resnext part
net = resnext_block(net, n, 16, 32, downsample=True)
net = resnext_block(net, 1, 32, 32, downsample=True)
net = resnext_block(net, n-1, 32, 32, downsample=True)
net = resnext_block(net, 1, 64, 32, downsample=True)
net = resnext_block(net, n-1, 64, 32, downsample=True)

net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, b.CLASS_3_NUMBER, activation='softmax')
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=b.PATH + 'log')
model.load(b.PATH + last_model_file)

predictions = []
for i in range(0, test_batches_no):
    testX, product_id, Y3, Y4 = b.load_batch(i, False)
    size = len(testX) / 10
    for i in range(0, 10):
        testXpart = testX[i * size, min((i + 1) * size, len(testX))]
        predictions.extend(model.predict_label(testXpart))
with open(b.PATH + 'ALL' + str(n) + 'resNextPredictions.txt', 'wb') as pickleFile:
    pickle.dump(predictions, pickleFile)