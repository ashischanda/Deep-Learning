import bson #install pymongo to get all functions
import pickle
import numpy as np
import io
from skimage.data import imread

# Simple data processing
PATH = '/gpfs/scratch/tug73611/bigBatches/'
training_directory = 'train.bson'
testing_directory = 'test.bson'
cross_validation_file_X = 'cross_validation_X.p'
cross_validation_file_cate1Y = 'cross_validation_cate1Y.p'
cross_validation_file_cate2Y = 'cross_validation_cate1Y.p'
cross_validation_file_cate3Y = 'cross_validation_cate1Y.p'
CATE_1 = 'cate1_Y'
CATE_2 = 'cate2_Y'
CATE_3 = 'cate3_Y'
PRODUCT_FILE = "prod_id_Y"
CLASS_1_NUMBER = 49
CLASS_2_NUMBER = 483
CLASS_3_NUMBER = 5270

BATCH_SIZE = 10000
IMG_WIDTH = 180
IMG_HEIGHT = 180
CHANNELS = 3

category_dict = pickle.load(open(PATH + 'category_dict.p', 'rb'))

# Method parameter is String - return value is Integer
def getLabel1(categoryID):
    item = category_dict[categoryID]
    return item[0]

def getLabel2(categoryID):
    item = category_dict[categoryID]
    return item[1]

def getLabel3(categoryID):
    item = category_dict[categoryID]
    return item[2]

def divide_into_files(flag):
    batchX = []
    cate1Y = []
    cate2Y = []
    cate3Y = []
    cross_validation_X = []
    cross_validation_cate1Y = []
    cross_validation_cate2Y = []
    cross_validation_cate3Y = []
    if flag:
        filename = training_directory
    else:
        filename = testing_directory
    with open(PATH + filename, 'rb', buffering=True) as file:
        i = 0
        k = 0
        for line in bson.decode_file_iter(file):
            pictures = []
            for pic in line['imgs']:
                pictures.append(np.array(imread(io.BytesIO(pic['picture'])), dtype=np.float32))
            batchX.extend(pictures)
            if flag:
                category_id = int(line['category_id'])
                category_index1 = getLabel1(str(category_id))
                category_index2 = getLabel2(str(category_id))
                category_index3 = getLabel3(str(category_id))
                cate1Y.extend(np.full(len(pictures, ), category_index1, dtype=np.int32))
                cate2Y.extend(np.full(len(pictures, ), category_index2, dtype=np.int32))
                cate3Y.extend(np.full(len(pictures, ), category_index3, dtype=np.int32))
            else:
                product_id = int(line['_id'])
                cate3Y.append(np.full(len(pictures, ), product_id))
            i += 1
            if (i == BATCH_SIZE):
                if flag:
                    cross_validation_indices = list(np.random.choice(len(batchX), len(batchX) // 10000, replace=False))
                    l = 0
                    while l < len(batchX):
                        if l in cross_validation_indices:
                            cross_validation_X.append(batchX[l])
                            cross_validation_cate1Y.append(cate1Y[l])
                            cross_validation_cate2Y.append(cate2Y[l])
                            cross_validation_cate3Y.append(cate3Y[l])
                            del batchX[l]
                            del cate1Y[l]
                            del cate2Y[l]
                            del cate3Y[l]
                            cross_validation_indices.remove(l)
                        else:
                            l += 1
                    with open(PATH + CATE_1 + str(k) + filename, 'wb') as pickleFile:
                        pickle.dump(cate1Y, pickleFile)
                    with open(PATH + CATE_2 + str(k) + filename, 'wb') as pickleFile:
                        pickle.dump(cate2Y, pickleFile)
                    with open(PATH + CATE_3 + str(k) + filename, 'wb') as pickleFile:
                        pickle.dump(cate3Y, pickleFile)
                else:
                    with open(PATH + PRODUCT_FILE + str(k) + filename, 'wb') as pickleFile:
                        pickle.dump(cate3Y, pickleFile)
                with open(PATH + 'X' + str(k) + filename, 'wb') as pickleFile:
                    pickle.dump(batchX, pickleFile)
                    print(str(k) + ' batch length: ' + str(len(batchX)))
                k += 1
                i = 0
                batchX = []
                cate1Y = []
                cate2Y = []
                cate3Y = []
        with open(PATH + 'X' + str(k) + filename, 'wb') as pickleFile:
            pickle.dump(batchX, pickleFile)
            print(str(k) + ' batch length: ' + str(len(batchX)))
        if flag:
            with open(PATH + CATE_1 + str(k) + filename, 'wb') as pickleFile:
                pickle.dump(cate1Y, pickleFile)
            with open(PATH + CATE_2 + str(k) + filename, 'wb') as pickleFile:
                pickle.dump(cate2Y, pickleFile)
            with open(PATH + CATE_3 + str(k) + filename, 'wb') as pickleFile:
                pickle.dump(cate3Y, pickleFile)
            with open(PATH + cross_validation_file_X, 'wb') as pickleFile:
                pickle.dump(cross_validation_X, pickleFile)
                print('Cross validation length: ' + str(len(cross_validation_X)))
            with open(PATH + cross_validation_file_cate1Y, 'wb') as pickleFile:
                pickle.dump(cross_validation_cate1Y, pickleFile)
            with open(PATH + cross_validation_file_cate2Y, 'wb') as pickleFile:
                pickle.dump(cross_validation_cate2Y, pickleFile)
            with open(PATH + cross_validation_file_cate3Y, 'wb') as pickleFile:
                pickle.dump(cross_validation_cate3Y, pickleFile)
        else:
            with open(PATH + PRODUCT_FILE + str(k) + filename, 'wb') as pickleFile:
                pickle.dump(cate3Y, pickleFile)
        k += 1
    return k

def load_batch(batch_no, flag):
    Y1 = []
    Y2 = []
    if flag:
        filename = training_directory
        Y1 = pickle.load(open(PATH + CATE_1 + str(batch_no) + filename, 'rb'))
        Y2 = pickle.load(open(PATH + CATE_2 + str(batch_no) + filename, 'rb'))
        Y = pickle.load(open(PATH + CATE_3 + str(batch_no) + filename, 'rb'))
    else:
        filename = testing_directory
        Y = pickle.load(open(PATH + PRODUCT_FILE + str(batch_no) + filename, 'rb'))
    X = pickle.load(open(PATH + 'X' + str(batch_no) + filename, 'rb'))
    return X, Y, Y1, Y2

def load_cross_validation():
    X = pickle.load(open(PATH + cross_validation_file_X, 'rb'))
    Y1 = pickle.load(open(PATH + cross_validation_file_cate1Y, 'rb'))
    Y2 = pickle.load(open(PATH + cross_validation_file_cate2Y, 'rb'))
    Y3 = pickle.load(open(PATH + cross_validation_file_cate3Y, 'rb'))
    return X, Y1, Y2, Y3

def select_random_batches(batchesNo, sizeOfRandomSample=10):
    randombatches = np.random.choice(batchesNo, sizeOfRandomSample)
    return randombatches
