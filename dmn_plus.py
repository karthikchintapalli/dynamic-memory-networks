from __future__ import print_function
from __future__ import division

import sys
import time

import numpy as np
from copy import deepcopy

import tensorflow as tf
from attention_gru_cell import AttentionGRUCell

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

import babi_input

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 100
    embed_size = 80
    hidden_size = 80

    max_epochs = 30

    dropout = 0.9
    learning_rate = 0.001
    l2 = 0.001

    word2vec_init = False
    embedding_init = np.sqrt(3)

    num_hops = 3

    max_allowed_inputs = 130
    num_train = 9000

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1)/2) * (j - (ls - 1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

class DMN_PLUS(object):
    def load_data(self):
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)

        self.encoding = position_encoding(self.max_sen_len, self.config.embed_size)

    def init_placeholders(self):
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_sentences, self.max_sen_len))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def predict_answer(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.answer_placeholder))

        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss
        
    def training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs = optimizer.compute_gradients(loss)

        train_op = optimizer.apply_gradients(gvs)
        return train_op

    def q_repr(self):
        questions = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)

        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        h, q_vec = tf.nn.dynamic_rnn(gru_cell, questions, dtype=np.float32, sequence_length=self.question_len_placeholder)

        return q_vec

    def input_repr(self):
        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        outputs, h = tf.nn.bidirectional_dynamic_rnn(forward_gru_cell, backward_gru_cell, inputs, dtype=np.float32, sequence_length=self.input_len_placeholder)

        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs

    def attention(self, q_vec, prev_memory, fact_vec, reuse):
        with tf.variable_scope("attention", reuse=reuse):
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            att_weight = tf.contrib.layers.fully_connected(feature_vec, self.config.embed_size,
                            activation_fn=tf.nn.tanh, reuse=reuse, scope="fc1")

            att_weight = tf.contrib.layers.fully_connected(att_weight, 1,
                            activation_fn=None, reuse=reuse, scope="fc2")

        return att_weight

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        attentions = [tf.squeeze(
            self.attention(q_vec, memory, fact_vec, bool(hop_index) or bool(i)), axis=1)
            for i, fact_vec in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        if hop_index > 0:
            reuse = True
        else:
            reuse = False

        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            h, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config.hidden_size),
                    gru_inputs, dtype=np.float32,
                    sequence_length=self.input_len_placeholder)

        return episode

    def answer_module(self, rnn_output, q_vec):
        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                self.vocab_size, activation=None)

        return output

    def inference(self):
        # input module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            q_vec = self.q_repr()

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            fact_vecs = self.input_repr()

        self.attentions = []

        # episodic memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            memory = q_vec

            for i in range(self.config.num_hops):
                episode = self.generate_episode(memory, q_vec, fact_vecs, i)

                with tf.variable_scope("hop_%d" % i):
                    memory = tf.layers.dense(tf.concat([memory, episode, q_vec], 1),
                            self.config.hidden_size, activation=tf.nn.relu)

            output = memory

        # answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.answer_module(output, q_vec)

        return output

    def run_epoch(self, session, data, num_epoch=0, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config.batch_size
        total_loss = []
        accuracy = 0

        p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a = data
        qp, ip, ql, il, im, a = qp[p], ip[p], ql[p], il[p], im[p], a[p]

        for step in range(total_steps):
            index = range(step * config.batch_size,(step + 1) * config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            print pred
            answers = a[step*config.batch_size:(step+1)*config.batch_size]
            accuracy += np.sum(pred == answers)/float(len(answers))


            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.mean(total_loss), accuracy/float(total_steps)

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.init_placeholders()
        self.embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
        self.output = self.inference()
        self.pred = self.predict_answer(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()

