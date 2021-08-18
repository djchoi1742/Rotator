import argparse

license = """
Copyright ⓒ Dong Yul Oh, Kyong Joon Lee
Department of Radiology at Seoul National University Bundang Hospital. \n
If you have any question, please email us for assistance: dongyul.oh@snu.ac.kr \n """
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
                                 description='', epilog=license, add_help=False)

network_config = parser.add_argument_group('network setting (must be provided)')

network_config.add_argument('--data_path', type=str, dest='data_path', default='/data/SNUBH/Rotator/')
network_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp309')
network_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
network_config.add_argument('--batch_size', type=int, dest='batch_size', default=12)
network_config.add_argument('--numEpoch', type=int, dest='num_epoch', default=0)  # infinite loop
network_config.add_argument('--trial_serial', type=int, dest='trial_serial', default=1)
network_config.add_argument('--npy_name', type=str, dest='npy_name', default='exp309_trval_VIEW1.npy')
network_config.add_argument('--max_keep', type=int, dest='max_keep', default=20) # only use training
network_config.add_argument('--num_weight', type=int, dest='num_weight', default=5) # only use validation

parser.print_help()
config, unparsed = parser.parse_known_args()

import sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore SSE instruction warning on tensorflow

import tensorflow as tf
import numpy as np
import sklearn.metrics  # roc curve
import matplotlib.pyplot as plt
import pandas as pd
import re

trial_serial_str = '%03d' % config.trial_serial
log_path = os.path.join(config.data_path, 'ro_new', config.exp_name, 'logs-%s' % trial_serial_str)
result_path = os.path.join(config.data_path, 'ro_new', config.exp_name, 'result-%s' % trial_serial_str)
cam_path = os.path.join(result_path, 'cam')
ckpt_path = os.path.join(result_path, 'ckpt')
npy_path = os.path.join(config.data_path, 'ro_new', config.exp_name, 'data')

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(cam_path):
    os.makedirs(cam_path)

from setup_dataset2 import Dataset_v2

print(npy_path+'/'+config.npy_name)
dataset = Dataset_v2(data_dir=os.path.join(npy_path, config.npy_name), batch_size=config.batch_size)

from model import Network

model = Network(trainable=config.train)

from tensorflow_utils.Tensorboard import Tensorboard
from tensorflow_utils.Evaluation import *

#sess_config = tf.ConfigProto(log_device_placement=False)
#sess_config.gpu_options.allow_growth = True
#sess = tf.Session(config=sess_config)

# <Training Script>
# CUDA_VISIBLE_DEVICES=3 python main.py --train=true --exp_name=expn03 --batchSize=4 --numEpoch=0 --npy_name=trval_VIEW3.npy

# <Validation Script>
# CUDA_VISIBLE_DEVICES=3 python main.py --logDir=checkpoint --batchSize=4


def prep_ckpt_rename(prep_ckpt):
    return re.sub('/workspace/Rotator/', config.data_path, prep_ckpt)


def training():
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    tensorboard = Tensorboard(log_dir=log_path, overwrite=True)

    loss_rec = tf.get_variable(name='Loss', shape=[], trainable=False, initializer=tf.zeros_initializer(),
                               collections=['scalar'])
    auc_rec = tf.get_variable(name='AUC', shape=[], trainable=False, initializer=tf.zeros_initializer(),
                              collections=['scalar'])

    tensorboard.init_scalar(collections=['scalar'])
    tensorboard.init_images(collections=['images'], num_outputs=4)

    sess.run(tf.global_variables_initializer())
    # default graph (with global variables initialization)
    saver = tf.train.Saver(max_to_keep=config.max_keep)

    performance_per_epoch, max_perf_per_epoch = [], []
    while True:  # increment of training epochs
        sess.run([dataset.train.init_op, dataset.test.init_op])

        while True:  # increment of training steps
            try:
                images, labels = sess.run(dataset.train.next_batch)
            except tf.errors.OutOfRangeError:
                break
            feed_dict = {model.images: images, model.labels: labels, model.is_training: True}
            sess.run(model.train, feed_dict=feed_dict)

            current_step, current_epoch = sess.run([tf.train.get_global_step(), model.global_epoch])
            sys.stdout.write('Step: {0:>4d} ({1})\r'.format(current_step, current_epoch))

        sess.run(tf.assign_add(model.global_epoch, 1))
        # compute loss and logits for auc
        train_loss, logits = sess.run([model.loss, tf.nn.softmax(model.logits)], feed_dict=feed_dict)

        false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(labels, logits[:, 1], drop_intermediate=False)
        train_auc = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

        feed_dict.update({loss_rec: train_loss, auc_rec: train_auc})  # update for both scalar and image summary variable
        tensorboard.add_summary(sess=sess, feed_dict=feed_dict, log_type='train')

        # validation with roc-auc
        test_X, test_Y, test_loss_batch = [], [], []

        while True:  # increment of validation steps
            sys.stdout.write('Evaluation [{0}/{1}]\r'.format(len(test_loss_batch),
                                                             -(-dataset.test.data_length // config.batch_size)))
            try:
                images, labels = sess.run(dataset.test.next_batch)
            except tf.errors.OutOfRangeError:
                break
            if len(test_loss_batch) >= 100: break

            feed_dict = {model.images: images, model.labels: labels, model.is_training: False}
            test_loss, logits = sess.run([model.loss, tf.nn.softmax(model.logits)], feed_dict=feed_dict)

            test_X.extend(logits[:, 1])  # probability from softmax
            test_Y.extend(labels)
            test_loss_batch.append(test_loss)

        false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(test_Y, test_X, drop_intermediate=False)
        test_auc = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

        feed_dict.update({loss_rec: np.mean(test_loss_batch),
                          auc_rec: test_auc})  # update for both scalar and image summary variable
        tensorboard.add_summary(sess=sess, feed_dict=feed_dict, log_type='test')

        # display summary on tensorboard and console
        tensorboard.display_summary(time_stamp=True)

        # increment of training epoch
        current_epoch += 1
        # sess.run(model.global_epoch.assign(current_epoch))
        if current_epoch % 1 == 0:  # save checkpoint every 1 epochs
            performance_per_epoch.append(test_auc)
            performance_per_epoch = sorted(performance_per_epoch, reverse=True)
            #print ('test_auc: ', test_auc)
            if current_epoch < config.max_keep+1:
                max_perf_per_epoch.append(test_auc)
                max_perf_per_epoch = sorted(max_perf_per_epoch, reverse=True)
                saver.save(sess=sess, save_path=os.path.join(log_path, 'model.ckpt'),
                           global_step=current_step)
            elif test_auc > max_perf_per_epoch[-1]:
                max_perf_per_epoch.remove(max_perf_per_epoch[-1])
                max_perf_per_epoch.append(test_auc)
                max_perf_per_epoch = sorted(max_perf_per_epoch, reverse=True)
                saver.save(sess=sess, save_path=os.path.join(log_path, 'model.ckpt'),
                           global_step=current_step)
                #print ('current epoch saved: ', max_perf_per_epoch)

            #if test_auc == np.max(
            #        performance_per_epoch):  # early stop by picking the best performance among previous checkpoints
            #    saver.save(sess=sess, save_path=os.path.join(log_path, 'model.ckpt'))
            if current_epoch == config.num_epoch: break

    print('Training Complete...\n')
    sess.close()


def validation():
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    num_examples = len(dataset.id_test)
    roc_curve_name = '_'.join([config.npy_name, trial_serial_str, '%03d' % config.num_weight])+'.png'
    result_name = '_'.join([config.npy_name, trial_serial_str, '%03d' % config.num_weight]) + '.csv'
    output_name = '_'.join([config.npy_name, trial_serial_str, '%03d' % config.num_weight, 'output']) + '.csv'

    ckpt = tf.train.get_checkpoint_state(log_path)
    if not ckpt:
        raise ValueError('No checkpoint found in ' + log_path)
    #all_ckpt_paths = list(map(prep_ckpt_rename, ckpt.all_model_checkpoint_paths))

    weight_auc_csv = pd.read_csv(os.path.join(result_path,
                                              '_'.join([config.exp_name, trial_serial_str])+'.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('AUC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])
    all_ckpt_paths = list(map(lambda x: re.sub('/workspace/Rotator/', config.data_path, x), all_ckpt_paths))

    num_ckpt = len(all_ckpt_paths)
    print('num_ckpt: ', num_ckpt)

    outputs = np.zeros([num_ckpt, num_examples, model.class_num])
    probs = np.zeros([num_ckpt, num_examples, model.class_num])
    lbls = np.zeros([num_examples], dtype=np.int32)
    heatmaps = np.zeros([num_ckpt, num_examples,
                         model.image_height, model.image_width, model.image_channel])

    for ckpt_idx, ckpt_path in enumerate(all_ckpt_paths):
        print('Restoring: ' + ckpt_path)

        sess = tf.Session(config=sess_config)
        saver.restore(sess, ckpt_path)

        sess.run(dataset.test.init_op)

        test_X, test_Y = [], []

        num_iter = int(np.ceil(float(num_examples) / config.batch_size))
        step = 0

        while step < num_iter:
            sys.stdout.write('Evaluation [{0}/{1}]\r'.format(len(test_X) // config.batch_size,
                             -(-dataset.test.data_length // config.batch_size)))

            images, labels = sess.run(dataset.test.next_batch)

            feed_dict = {model.images: images, model.labels: labels, model.is_training: False}

            #test_logits = sess.run(tf.nn.softmax(model.logits), feed_dict=feed_dict)
            test_outputs, test_logits = sess.run([model.logits, tf.nn.softmax(model.logits)], feed_dict=feed_dict)
            heatmap = sess.run(model.grad_cam, feed_dict=feed_dict)

            outputs[ckpt_idx, step * config.batch_size:step * config.batch_size + len(labels)] = test_outputs
            probs[ckpt_idx, step*config.batch_size:step*config.batch_size+len(labels)] = test_logits
            heatmaps[ckpt_idx, step*config.batch_size:step*config.batch_size+len(labels)] = heatmap

            if ckpt_idx == 0:
                lbls[step*config.batch_size:step*config.batch_size+len(labels)] = labels

            test_X.extend(test_logits)
            step += 1

        sess.close()

    outputs = np.mean(outputs, axis=0)
    probs = np.mean(probs, axis=0)

    heatmaps = np.mean(heatmaps, axis=0)
    id_test = dataset.id_test

    output_0, output_1 = np.array(outputs)[:, 0], np.array(outputs)[:, 1]
    prob_0, prob_1 = np.array(probs)[:, 0], np.array(probs)[:, 1]
    result_csv = pd.DataFrame({'NUMBER': id_test, 'PROB_0': prob_0, 'PROB_1': prob_1, 'LABEL': lbls})
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)

    output_csv = pd.DataFrame({'NUMBER': id_test, 'OUTPUT_0': output_0, 'OUTPUT_1': output_1, 'LABEL': lbls})
    output_csv.to_csv(os.path.join(result_path, output_name), index=False)
    optimal_cutoff = draw_roc_curve(probs, lbls, target_index=1,
                                    name=os.path.join(result_path, roc_curve_name))

    print('Validation Complete...\n')


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()