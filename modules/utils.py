
import tensorflow as tf

def weight_from_caffe(caffenet):
    def func(shape, dtype):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        print 'init: ', name, shape, caffenet.params[name][0].data.shape
        return tf.transpose(caffenet.params[name][0].data, perm=[2 ,3 ,1 ,0])
    return func

def bias_from_caffe(caffenet):
    def func(shape, dtype):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        return caffenet.params[name][1].data
    return func


