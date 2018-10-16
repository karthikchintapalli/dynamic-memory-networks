from __future__ import print_function
from __future__ import division

import tensorflow as tf

import sys
import time
import argparse
import os

from dmn_plus import Config
from dmn_plus import DMN_PLUS


config = Config()
config.babi_id = sys.argv[1] if sys.argv[1] is not None else str(1)
config.l2 = 0.001
config.strong_supervision = False

print('training on babi task', config.babi_id)

best_overall_val_loss = float('inf')

with tf.variable_scope('DMN') as scope:
    model = DMN_PLUS(config)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    best_val_epoch = 0
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    best_val_accuracy = 0.0

    for epoch in range(config.max_epochs):
        print('Epoch {}'.format(epoch))
        start = time.time()

        train_loss, train_accuracy = model.run_epoch(
            session, model.train, epoch,
            train_op=model.train_step, train=True)
        valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
        print('Training loss: {}'.format(train_loss))
        print('Validation loss: {}'.format(valid_loss))
        print('Training accuracy: {}'.format(train_accuracy))
        print('Validation accuracy: {}'.format(valid_accuracy))

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_val_epoch = epoch
            if best_val_loss < best_overall_val_loss:
                best_overall_val_loss = best_val_loss
                best_val_accuracy = valid_accuracy
                saver.save(session, 'weights/task' + str(model.config.babi_id) + '.weights')

        prev_epoch_loss = train_loss
        print('epoch time: {}'.format(time.time() - start))

    print('Best validation accuracy:', best_val_accuracy)
