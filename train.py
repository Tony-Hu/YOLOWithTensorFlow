import datetime
from VOC2012ImageReader import ImageReader
from YOLONN import YoloNeuralNetwork
import tensorflow as tf
import time
import sys
import os

class Trainer:
    if sys.platform =='win32':
        OUT_PUT_DIR = '\\report'
    else:
        OUT_PUT_DIR = '/report'

    if sys.platform == 'win32':
        BOARD_OUT_PUT_DIR = '\\board'
    else:
        BOARD_OUT_PUT_DIR = '/board'

    def __init__(self, max_iter=30000, summary_iter=10, initial_learning_rate=0.001, decay_steps=30000, save_iter=100):

        # Load data sets
        self.training_data_set, self.validation_data_set, self.testing_data_set = ImageReader.read()
        self.training_iterator = self.training_data_set.make_one_shot_iterator()

        # Load network module
        self.yolo_nn = YoloNeuralNetwork()

        self.max_iter = max_iter
        self.summary_iter = summary_iter
        self.save_iter = save_iter
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps

        self.ckpt_file = os.path.join(Trainer.OUT_PUT_DIR, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()
        print(self.summary_op)

        if not os.path.exists(Trainer.BOARD_OUT_PUT_DIR):
            os.makedirs(Trainer.BOARD_OUT_PUT_DIR)
        self.writer = tf.summary.FileWriter(Trainer.BOARD_OUT_PUT_DIR, flush_secs=60)

        # Set up learning rate
        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            0.1, False, name='learning_rate')

        # Optimize Gradient descent
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(
            self.yolo_nn.total_loss, global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        # Starting session
        gpu_memory = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_memory))
        self.sess.run(tf.global_variables_initializer())
        self.variable_to_restore = tf.global_variables()
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)

    def train(self):
        for step in range(self.max_iter):
            load_starting_time = time.time()
            images, labels = self.training_iterator.get_next()
            images = self.sess.run(images)
            labels = self.sess.run(labels)
            load_end_time = time.time()
            feed_dict = {self.yolo_nn.images: images, self.yolo_nn.labels: labels}

            train_starting_time = time.time()
            summary_str, loss, _ = self.sess.run(
                [self.summary_op, self.yolo_nn.total_loss, self.train_op],
                feed_dict=feed_dict)
            train_end_time = time.time()
            self.writer.add_summary(summary_str, step)

            if step % self.summary_iter == 0:
                log_str = ('{} Step: {}, Learning rate: {},'
                           ' Loss: {:5.3f}\nTraining Speed: {:.3f}s/iter,'
                           ' Load: {:.3f}s/iter, Remain: {}').format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'), int(step),
                    round(self.learning_rate.eval(session=self.sess), 6),
                    loss,
                    train_end_time - train_starting_time,
                    load_end_time - load_starting_time,
                    self.max_iter - step)
                print(log_str)

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    Trainer.OUT_PUT_DIR))
                self.saver.save(self.sess, self.ckpt_file,
                                global_step=self.global_step)