import os
import time
import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import scipy.misc
import random
import subprocess
from datetime import datetime
from math import ceil

# from modules import BasicConvLSTMCell
# from modules.model_easyflow import *
from modules.videosr_ops_lite import *

os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

DATA_TEST='./data/test/calendar'
# DATA_TEST='./data/test/hitachi_isee5_001'
DATA_TRAIN='./data/train/'


class VIDEOSR(object):
    def __init__(self):
        self.num_frames = 3
        self.scale_factor = 4

    def test(self, dataPath=None, scale_factor=4, num_frames=3):

        import scipy.misc
        dataPath = DATA_TEST
        inList = sorted(glob.glob(os.path.join(dataPath, 'input{}/*.png').format(scale_factor)))
        inp = [scipy.misc.imread(i).astype(np.float32) / 255.0 for i in inList]
        # inp = [scipy.misc.imresize(i, [120, 160]) / 255.0 for i in inp]
        inp = [i[:120, :160, :] for i in inp]

        print 'Testing path: {}'.format(dataPath)
        print '# of testing frames: {}'.format(len(inList))

        DATA_TEST_OUT = DATA_TEST+'_SR_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(DATA_TEST_OUT)
        
        cnt = 0
        self.scale_factor = scale_factor
        reuse = False
        for idx0 in xrange(len(inList)):
            cnt += 1
            T = num_frames / 2

            imgs = [inp[0] for i in xrange(idx0 - T, 0)]
            imgs.extend([inp[i] for i in xrange(max(0, idx0 - T), idx0)])
            imgs.extend([inp[i] for i in xrange(idx0, min(len(inList), idx0 + T + 1))])
            imgs.extend([inp[-1] for i in xrange(idx0 + T, len(inList) - 1, -1)])

            dims = imgs[0].shape
            if len(dims) == 2:
                imgs = [np.expand_dims(i, -1) for i in imgs]
            h, w, c = imgs[0].shape
            out_h = h * scale_factor
            out_w = w * scale_factor
            padh = int(ceil(h / 4.0) * 4.0 - h)
            padw = int(ceil(w / 4.0) * 4.0 - w)
            imgs = [np.pad(i, [[0, padh], [0, padw], [0, 0]], 'edge') for i in imgs]
            imgs = np.expand_dims(np.stack(imgs, axis=0), 0)

            if idx0 == 0:
                frames_lr = tf.placeholder(dtype=tf.float32, shape=imgs.shape)
                frames_ref_ycbcr = rgb2ycbcr(frames_lr[:, T:T + 1, :, :, :])
                frames_ref_ycbcr = tf.tile(frames_ref_ycbcr, [1, num_frames, 1, 1, 1])

                with open('spmc_120_160_4x3f.pb', 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    output = tf.import_graph_def(graph_def, input_map={'Placeholder:0': frames_lr}, return_elements=['output:0'])
                    output = output[0]
                    print(output.get_shape())

                if len(dims) == 3:
                    output_rgb = ycbcr2rgb(tf.concat([output, resize_images(frames_ref_ycbcr,
                                                                            [(h + padh) * scale_factor,
                                                                             (w + padw) * scale_factor],
                                                                            method=2)[:, :, :, :, 1:3]], -1))
                else:
                    output_rgb = output
                output = output[:, :, :out_h, :out_w, :]
                output_rgb = output_rgb[:, :, :out_h, :out_w, :]
    
            if cnt == 1:
                sess = tf.Session()
                reuse = True

            case_path = dataPath.split('/')[-1]
            print 'Testing - ', case_path, len(imgs)
            [imgs_hr, imgs_hr_rgb] = sess.run([output, output_rgb], feed_dict={frames_lr: imgs})
            
            scipy.misc.imsave(os.path.join(DATA_TEST_OUT, 'y_%03d.png'%(idx0)),
                              im2uint8(imgs_hr[0, -1, :, :, 0]))
            if len(dims) == 3:
                scipy.misc.imsave(os.path.join(DATA_TEST_OUT, 'rgb_%03d.png'%(idx0)),
                                  im2uint8(imgs_hr_rgb[0, -1, :, :, :]))
        print 'SR results path: {}'.format(DATA_TEST_OUT)


def main(_):
    model = VIDEOSR()
    model.test()

if __name__ == '__main__':
    tf.app.run()
