from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tarfile
import urllib
import fashion_input
import fashion

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


def test():
    data = input_data.read_data_sets('/home/weili/PycharmProjects/fashion/data/fashion')
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('/home/weili/PycharmProjects/fashion/data/model.ckpt-19999.meta')

        # graph = tf.get_default_graph()
        data_ = data.test.next_batch(FLAGS.batch_size)
        x = data_[0]
        y = data_[1]
        x_change = np.reshape(x, (100, 28, 28, -1))
        y_np = fashion.inference(x_change,100)
        # y_np = tf.arg_max(y_np,1)

        saver =tf.train.Saver()
        saver.restore(sess, '/home/weili/PycharmProjects/fashion/data/model.ckpt-299')

        # tf.image_summary('x', x_change)
        # summary_op = tf.merge_all_summaries()

        print('f1:')
        print(y)
        print('f2:')
        print(sess.run(y_np))

        # print(sess.run(y_np))
        # tf.Print(tf.argmax(y_np,1))
        # print(graph.get_all_collection_keys())
        # print(graph.get_operations())
        # input_x = graph.get_operation_by_name('test_images').outputs[0]
        # y_np = sess.run(y,feed_dict = {x_d:x_change})
        # print("Actual class: ", str(sess.run(tf.argmax(y, 1))), \
        #   ", predict class ",str(sess.run(tf.argmax(y_np, 1))), \
        #   ", predict ", str(sess.run(tf.equal(tf.argmax(y, 1), tf.argmax(y_np, 1))))
        #   )





