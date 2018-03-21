import tensorflow as tf
import batches as b
import pickle

def calculate_params_no(netname):
    with open(b.PATH + netname + 'parametersNo.txt', 'wb') as pickleFile:
        data = {}
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            data['Shape'] = shape
            data['lenOfShape'] = len(shape)
            variable_parameters = 1
            for dim in shape:
                data['dimensions'] = dim
                variable_parameters *= dim.value
            data['variable_parameters'] = variable_parameters
            total_parameters += variable_parameters
        data['Total_parameters'] = total_parameters
        pickle.dump(data, pickleFile)