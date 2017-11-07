import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def get_shape(x):
    shape = tf.shape(x)
    check = tf.Assert(tf.reduce_all(shape >= 0), ["EASYFLOW: Need value.shape >= 0, got ", shape])
    shape = control_flow_ops.with_dependencies([check], shape)
    return [shape[i] for i in xrange(shape.shape.as_list()[0])]


def zero_upsampling(x, scale_factor):
    dims = x.get_shape().as_list()
    if len(dims) == 5:
        n, t, h, w, c = dims
        y = tf.concat([x] + [tf.zeros_like(x)] * (scale_factor ** 2 - 1), -1)
        y = tf.reshape(y, [n, t, h, w, scale_factor, scale_factor, c])
        y = tf.transpose(y, [0, 1, 2, 4, 3, 5, 6])
        y = tf.reshape(y, [n, t, h * scale_factor, w * scale_factor, c])
    elif len(dims) == 4:
        n, h, w, c = dims
        y = tf.concat([x] + [tf.zeros_like(x)] * (scale_factor ** 2 - 1), -1)
        y = tf.reshape(y, [n, h, w, scale_factor, scale_factor, c])
        y = tf.transpose(y, [0, 1, 3, 2, 4, 5])
        y = tf.reshape(y, [n, h * scale_factor, w * scale_factor, c])
    return y


def leaky_relu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)


def prelu(x):
    alphas = tf.get_variable('alpha', x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5

    return pos + neg


def display_tf_variables(train_vars):
    print 'Training Variables: '
    for var in train_vars:
        print '\t', var.name


def resize_images(images, size, method=2, align_corners=False):
    dims = len(images.get_shape())
    if dims == 5:
        n, t, h, w, c = images.get_shape().as_list()
        images = tf.reshape(images, [n * t, h, w, c])
    images = tf.image.resize_images(images, size, method, align_corners)
    if dims == 5:
        images = tf.reshape(images, [n, t, size[0], size[1], c])
    return images


def rgb2y(inputs):
    with tf.name_scope('rgb2y'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2y input should be RGB or grayscale!'
        dims = len(inputs.get_shape())
        if dims == 4:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0
        elif dims == 5:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 1, 3]) / 255.0
        output = tf.reduce_sum(inputs * scale, reduction_indices=dims - 1, keep_dims=True)
        output = output + 16 / 255.0
    return output


def rgb2ycbcr(inputs):
    with tf.name_scope('rgb2ycbcr'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
        ndims = len(inputs.get_shape())
        origT = [[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]]
        origOffset = [16.0, 128.0, 128.0]
        if ndims == 4:
            origT = [tf.reshape(origT[i], [1, 1, 1, 3]) / 255.0 for i in xrange(3)]
        elif ndims == 5:
            origT = [tf.reshape(origT[i], [1, 1, 1, 1, 3]) / 255.0 for i in xrange(3)]
        output = []
        for i in xrange(3):
            output.append(tf.reduce_sum(inputs * origT[i], reduction_indices=-1, keep_dims=True) + origOffset[i] / 255.0)
        return tf.concat(output, -1)


def ycbcr2rgb(inputs):
    with tf.name_scope('ycbcr2rgb'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
        ndims = len(inputs.get_shape())
        # origT = np.array([[65.481, 128.553, 24.966], [-37.797 -74.203 112], [112 -93.786 -18.214]])
        # T = tf.inv(origT)
        Tinv = [[0.00456621, 0., 0.00625893], [0.00456621, -0.00153632, -0.00318811], [0.00456621, 0.00791071, 0.]]
        origOffset = [16.0, 128.0, 128.0]
        if ndims == 4:
            origT = [tf.reshape(Tinv[i], [1, 1, 1, 3]) * 255.0 for i in xrange(3)]
            origOffset = tf.reshape(origOffset, [1, 1, 1, 3]) / 255.0
        elif ndims == 5:
            origT = [tf.reshape(Tinv[i], [1, 1, 1, 1, 3]) * 255.0 for i in xrange(3)]
            origOffset = tf.reshape(origOffset, [1, 1, 1, 1, 3]) / 255.0
        output = []
        for i in xrange(3):
            output.append(tf.reduce_sum((inputs - origOffset) * origT[i], reduction_indices=-1, keep_dims=True))
        return tf.concat(output, -1)
    

def rgb2gray(inputs):
    with tf.name_scope('rgb2gray'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2y input should be RGB or grayscale!'
        dims = len(inputs.get_shape())
        if dims == 4:
            scale = tf.reshape([0.299, 0.587, 0.114], [1, 1, 1, 3])
        elif dims == 5:
            scale = tf.reshape([0.299, 0.587, 0.114], [1, 1, 1, 1, 3])
        output = tf.reduce_sum(inputs * scale, reduction_indices=dims - 1, keep_dims=True)
    return output
