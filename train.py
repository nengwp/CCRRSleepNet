#! /usr/bin/python
# -*- coding: utf8 -*-
'''
https://github.com/akaraspt/deepsleepnet
Copyright 2017 Akara Supratak and Hao Dong.  All rights reserved.
'''


import os

import numpy as np
import tensorflow as tf

from ccrrsleep.trainer import CCRRFeatureNetTrainer, CCRRSleepNetTrainer
from ccrrsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../data/data-39/eeg_fpz_cz',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('n_folds', 20,
                           """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 17,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('pretrain_epochs', 30,
                            """Number of epochs for pretraining CCRRFeatureNet.""")
tf.app.flags.DEFINE_integer('finetune_epochs', 80,
                            """Number of epochs for fine-tuning CCRRSleepNet.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume the training process.""")


def pretrain(n_epochs):
    trainer = CCRRFeatureNetTrainer(
        data_dir=FLAGS.data_dir, 
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds, 
        fold_idx=FLAGS.fold_idx,
        batch_size=256,
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        interval_plot_filter=50,
        interval_save_model=50,
        interval_print_cm=1
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return pretrained_model_path


def finetune(model_path, n_epochs):
    trainer = CCRRSleepNetTrainer(
        data_dir=FLAGS.data_dir, 
        output_dir=FLAGS.output_dir, 
        n_folds=FLAGS.n_folds, 
        fold_idx=FLAGS.fold_idx, 
        batch_size=10, 
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=2,
        return_last=False,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path, 
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return finetuned_model_path


def main(argv=None):
    # Output dir
    output_dir = os.path.join(FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))
    if not FLAGS.resume:
        if tf.gfile.Exists(output_dir):
            tf.gfile.DeleteRecursively(output_dir)
        tf.gfile.MakeDirs(output_dir)

    pretrained_model_path = pretrain(
        n_epochs=FLAGS.pretrain_epochs
    )

    finetuned_model_path = finetune(
        model_path=pretrained_model_path, 
        n_epochs=FLAGS.finetune_epochs
    )


def make_print_to_file(path='./',fileName=None):
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            # self.log = open(os.path.join(path, filename), "a", encoding='utf8', )
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    if not fileName:
        fileName = datetime.datetime.now().strftime('log_' + '%Y_%m_%d_%Hh_%Mmin')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, ' '))


if __name__ == "__main__":
    np.random.seed(1)
    tf.set_random_seed(1)
    log_path = './log'
    os.makedirs(log_path, exist_ok=True)
    make_print_to_file(log_path,"fold_{}_".format(FLAGS.fold_idx))
    tf.compat.v1.app.run()
    # tf.compat.v1.app.run(main())