from __future__ import division, print_function, absolute_import

import tflearn
from layers.wideresnet_block import wideresnet_block
import batches as b
import pickle
from paramsNo import calculate_params_no

with open(b.PATH + 'ALLWideResNetEntered.txt', 'wb') as pickleFile:
    pickle.dump([], pickleFile)
# Number of wideResNet blocks
n = 2
epochs_no = 10 # 100

k = 2 # width; according to paper this is best size; tested values: 2, 4, 6, 8, 10, 12

# training_batches_no = b.divide_into_files(True)
# test_batches_no = b.divide_into_files(False)
total_training_batches_no = 707
training_batches_no = 707 # put 707 to work with all data
total_test_batches_no = 177
test_batches_no = 5 # put 177 to work with all data

cvX, cvY1, cvY2, cvY3 = b.load_cross_validation()
cvY = tflearn.data_utils.to_categorical(cvY3, b.CLASS_3_NUMBER) # put appropriate Y value here depending on the category you want to run for
with open(b.PATH + 'ALLWideResNetCross validation finished.txt', 'wb') as pickleFile:
    pickle.dump([], pickleFile)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
with open(b.PATH + 'ALLWideResNetPreprocessing finished.txt', 'wb') as pickleFile:
    pickle.dump([], pickleFile)

# Building Residual Network
net = tflearn.input_data(shape=[None, b.IMG_WIDTH, b.IMG_HEIGHT, b.CHANNELS],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)

#wideresnet part
net = wideresnet_block(net, n, 16, k, downsample=True)
net = wideresnet_block(net, 1, 32, k, downsample=True)
net = wideresnet_block(net, n-1, 32, k, downsample=True)
net = wideresnet_block(net, 1, 64, 2, downsample=True)
net = wideresnet_block(net, n-1, 64, k, downsample=True)

net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, b.CLASS_3_NUMBER, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
with open(b.PATH + 'ALLWideResNetNet created.txt', 'wb') as pickleFile:
    pickle.dump([], pickleFile)

# Training
model = tflearn.DNN(net, tensorboard_verbose=0, clip_gradients=0., tensorboard_dir=b.PATH + 'log')
for j in range(0, epochs_no):
    with open(b.PATH + 'ALL' + str(j) + 'wideResNetEpoch.txt', 'wb') as pickleFile:
        pickle.dump([], pickleFile)
    validation_set = None
    snapshot_epoch = False
    for i in range(0, training_batches_no):
        with open(b.PATH + 'ALL' + str(i) + 'WideResNetBatch.txt', 'wb') as pickleFile:
            pickle.dump([], pickleFile)
        X, Y, Y1, Y2 = b.load_batch(i, True)
        Y = tflearn.data_utils.to_categorical(Y, b.CLASS_3_NUMBER)
        if i == training_batches_no - 1 or i % 50 == 0:
            validation_set = cvX, cvY
            snapshot_epoch = True
        model.fit(X, Y, n_epoch=1, validation_set=validation_set, snapshot_epoch=snapshot_epoch, show_metric=True, run_id='wideresnet')
        if i % 50 == 0:
            model.save(b.PATH + 'ALLmodel_wideresnet_batch_' + str(i) + 'epoch_' + str(j) + '.txt')

    model.save(b.PATH + 'ALLmodel_wideresnet_epoch_' + str(j) + '.txt')
with open(b.PATH + 'ALLWideResNetModelTrained.txt', 'wb') as pickleFile:
    pickle.dump([], pickleFile)

predictions = []
random_test_batches = b.select_random_batches(total_test_batches_no, test_batches_no)
for i in range(0, test_batches_no):
    k = random_test_batches[i]
    with open(b.PATH + 'ALL' + str(k) + 'TestWideResNetBatch.txt', 'wb') as pickleFile:
        pickle.dump([], pickleFile)
    testX, product_id, Y3, Y4 = b.load_batch(k, False)
    #cut each batch into 10 parts
    predictions.extend(model.predict_label(testX))
with open(b.PATH + 'ALL' + str(n) + 'wideResNetPredictions.txt', 'wb') as pickleFile:
    pickle.dump(predictions, pickleFile)
with open(b.PATH + 'ALLResNextPredictionDone.txt', 'wb') as pickleFile:
    pickle.dump([], pickleFile)

calculate_params_no('wideResNet')
with open(b.PATH + 'ALLWideResNetParametersCalculated.txt', 'wb') as pickleFile:
    pickle.dump([], pickleFile)