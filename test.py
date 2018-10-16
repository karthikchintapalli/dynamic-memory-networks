from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import sys
import time
import argparse

from dmn_plus import Config
from dmn_plus import DMN_PLUS

config = Config()

config.babi_id = sys.argv[1]

config.strong_supervision = False
config.train_mode = False

print('testing on babi task', config.babi_id)

with tf.variable_scope('DMN') as scope:
    model = DMN_PLUS(config)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)
    saver.restore(session, 'weights/task' + str(model.config.babi_id) + '.weights')

    test_loss, test_accuracy = model.run_epoch(session, model.test)

    print('')
    print('Test accuracy:', test_accuracy)
