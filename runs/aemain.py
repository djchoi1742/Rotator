import argparse

license = """
Copyright ⓒ Dong Yul Oh, Kyong Joon Lee
Department of Radiology at Seoul National University Bundang Hospital. \n
If you have any question, please email us for assistance: dongyul.oh@snu.ac.kr \n """
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
                                 description='', epilog=license, add_help=False)

network_config = parser.add_argument_group('network setting (must be provided)')

network_config.add_argument('--data_path', type=str, dest='data_path', default='/data/SNUBH/Rotator/')
network_config.add_argument('--pre_name', type=str, dest='pre_name', default='exp308')
network_config.add_argument('--exp_name', type=str, dest='exp_name', default='aggr02')
network_config.add_argument('--model_name', type=str, dest='model_name', default='network')
network_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
network_config.add_argument('--batch_size', type=int, dest='batch_size', default=12)
network_config.add_argument('--numEpoch', type=int, dest='num_epoch', default=0)  # infinite loop
network_config.add_argument('--trial_serial', type=int, dest='trial_serial', default=30)
network_config.add_argument('--npy_name', type=str, dest='npy_name', default='bigc_all.npy')
network_config.add_argument('--max_keep', type=int, dest='max_keep', default=3)
network_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)
network_config.add_argument('--num_param', type=int, dest='num_param', default=16)
network_config.add_argument('--act_fn', type=str, dest='act_fn', default='relu')

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

trial_serial_str = '%03d' % (config.trial_serial)
pre_path = os.path.join(config.data_path, 'ro_new', config.pre_name)
log_path = os.path.join(config.data_path, 'ro_new', config.exp_name,
                        'logs-%s' % (trial_serial_str), config.model_name,
                        'param-%03d' % (config.num_param)+'_'+config.act_fn)
result_path = os.path.join(config.data_path, 'ro_new', config.exp_name, 'result-%s' % (trial_serial_str),
                           config.model_name)
cam_path = os.path.join(result_path, 'cam')
ckpt_path = os.path.join(result_path, 'ckpt')

npy_path = os.path.join(config.data_path, 'ro_new', config.exp_name, 'data')

if not os.path.exists(result_path): os.makedirs(result_path)
if not os.path.exists(cam_path): os.makedirs(cam_path)

from setup_adataset import Dataset_npy


# dataset setting
print('**npy file: ', npy_path+'/'+config.npy_name)
dataset = Dataset_npy(data_dir=os.path.join(npy_path, config.npy_name), batch_size=config.batch_size,
                      only_test=bool(1 - config.train))


from apmodel import *
from model import *

import apmodel
from model import Network

# model, prev_model setting
infer_name = 'Aggre_' + config.model_name
model = getattr(apmodel, infer_name)(num_param=int(config.num_param), act_fn=str(config.act_fn),
                                     trainable=config.train)
print('**model name: ', infer_name)
#model = Aggre_network(num_param=int(config.num_param), act_fn=str(config.act_fn), trainable=config.train)
prev_model = Network(trainable=config.train)


pretrain_vars = list(set(get_prep_conv_vars()))
global_vars = tf.global_variables()
new_vars = filter(lambda a: not a in pretrain_vars, global_vars)

from tensorflow_utils.Tensorboard import Tensorboard
from tensorflow_utils.Evaluation import *


def prep_ckpt_rename(prep_ckpt):
    return re.sub('/workspace/Rotator/', config.data_path, prep_ckpt)


def prep_weight_process(sess, pre_path, prep_saver, imgs, id, height, width, class_num):
    conv = []
    for idx in range(1, 4):
        prep_ckpt = tf.train.get_checkpoint_state(os.path.join(pre_path,
                                                               'logs-%03d'%idx))
        if not prep_ckpt:
            raise ValueError('No checkpoint found in '+os.path.join(pre_path,
                                                                    'logs-%03d' % idx))
        weight_auc_path = os.path.join(pre_path, 'result-%03d'%idx)
        weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path,
                                                  '_'.join([config.pre_name, '%03d'%idx])+'.csv'))
        weight_auc_csv = weight_auc_csv.sort_values('AUC', ascending=False)
        prep_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])
        prep_ckpt_paths = list(map(lambda x: re.sub('/workspace/Rotator/', config.data_path, x), prep_ckpt_paths))
        num_prep_ckpt = len(prep_ckpt_paths)

        convs = np.zeros([num_prep_ckpt, len(id),
                          height, width, class_num])
        for prep_ckpt_idx, prep_ckpt_path in enumerate(prep_ckpt_paths):
            prep_saver.restore(sess, prep_ckpt_path)
            prep_conv = sess.run(prev_model.features_2D,
                                 feed_dict={prev_model.images: imgs[idx], prev_model.is_training: False})
            convs[prep_ckpt_idx, 0:len(id)] = prep_conv
        convs = np.mean(convs, axis=0)
        conv.append(convs)
    return conv


def training():
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    prep_saver = tf.train.Saver(var_list=pretrain_vars)
    saver = tf.train.Saver(max_to_keep=config.max_keep)

    init_op = tf.group(tf.variables_initializer(var_list=new_vars),
                       tf.local_variables_initializer())

    sess = tf.Session(config=sess_config)
    sess.run(init_op)

    tensorboard = Tensorboard(log_dir=log_path, overwrite=True)
    loss_rec = tf.get_variable(name='Loss', shape=[], trainable=False, initializer=tf.zeros_initializer(),
                               collections=['scalar'])
    auc_rec = tf.get_variable(name='AUC', shape=[], trainable=False, initializer=tf.zeros_initializer(),
                              collections=['scalar'])

    tensorboard.init_scalar(collections=['scalar'])

    performance_per_epoch, max_perf_per_epoch = [], []

    while True:
        sess.run([dataset.train.init_op, dataset.test.init_op])
        #step = 0
        while True:
            try:
                img1, img2, img3, lbl, id, age, sxf, sxm, vas, tm0, tm1, tm2, dm0, dm1, dm2 = \
                sess.run(dataset.train.next_batch)

                imgs = dict({1: img1, 2: img2, 3: img3})
                conv = prep_weight_process(sess, pre_path, prep_saver, imgs, id,
                                           model.image_height, model.image_width, model.class_num)

            except tf.errors.OutOfRangeError:
                break

            feed_dict = {model.images1: conv[0], model.images2: conv[1], model.images3: conv[2],
                         model.labels: lbl, model.is_training: True,
                         model.ages: age, model.sxf: sxf, model.sxm: sxm,
                         model.vas: vas, model.tm0: tm0, model.tm1: tm1, model.tm2: tm2,
                         model.dm0: dm0, model.dm1: dm1, model.dm2: dm2,
                         prev_model.images: np.zeros([10, prev_model.image_height, prev_model.image_width, 1]),
                         prev_model.labels: np.zeros([len(id)]), prev_model.is_training: True
                        }

            sess.run(model.train, feed_dict=feed_dict)

            current_step, current_epoch = sess.run([tf.train.get_global_step(), model.global_epoch])

            sys.stdout.write('Step: {0:>4d} ({1})\r'.format(current_step, current_epoch))

        sess.run(tf.assign_add(model.global_epoch, 1))

        if config.model_name != 'logistic':
            train_loss, logits = sess.run([model.loss, tf.nn.softmax(model.logits)], feed_dict=feed_dict)
            pos_prob = logits[:,1]
        else:
            train_loss, logits = sess.run([model.loss, model.logits], feed_dict=feed_dict)
            pos_prob = logits
        #import pdb; pdb.set_trace()

        false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(lbl, pos_prob, drop_intermediate=False)
        train_auc = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

        feed_dict.update({loss_rec: train_loss, auc_rec: train_auc})  # update for both scalar and image summary variable
        tensorboard.add_summary(sess=sess, feed_dict=feed_dict, log_type='train')

        # validation with roc-auc
        test_X, test_Y, test_loss_batch = [], [], []
        num_examples = len(dataset.id_test)

        num_iter = int(np.ceil(float(num_examples) / config.batch_size))
        step = 0

        while step < num_iter:  # increment of validation steps
            sys.stdout.write('Evaluation [{0}/{1}]\r'.format(len(test_loss_batch),
                                                             -(-dataset.test.data_length // config.batch_size)))
            #try:
            img1, img2, img3, lbl, id, age, sxf, sxm, vas, tm0, tm1, tm2, dm0, dm1, dm2 = \
            sess.run(dataset.test.next_batch)

            imgs = dict({1: img1, 2: img2, 3: img3})

            conv = prep_weight_process(sess, pre_path, prep_saver, imgs, id,
                                       model.image_height, model.image_width, model.class_num)

            feed_dict = {model.images1: conv[0], model.images2: conv[1], model.images3: conv[2],
                         model.labels: lbl, model.is_training: False,
                         model.ages: age, model.sxf: sxf, model.sxm: sxm,
                         model.vas: vas, model.tm0: tm0, model.tm1: tm1, model.tm2: tm2,
                         model.dm0: dm0, model.dm1: dm1, model.dm2: dm2,
                         prev_model.images: np.zeros([10, prev_model.image_height, prev_model.image_width, 1]),
                         prev_model.labels: np.zeros([len(id)]), prev_model.is_training: False
                         }

            if len(test_loss_batch) >= 100: break

            if config.model_name != 'logistic':
                test_loss, logits = sess.run([model.loss, tf.nn.softmax(model.logits)], feed_dict=feed_dict)
                test_X.extend(logits[:, 1])
            else:
                test_loss, logits = sess.run([model.loss, model.logits], feed_dict=feed_dict)
                test_X.extend(logits)

            #test_loss, logits = sess.run([model.loss, tf.nn.softmax(model.logits)], feed_dict=feed_dict)

            #test_X.extend(logits[:, 1])  # probability from softmax
            test_Y.extend(lbl)
            test_loss_batch.append(test_loss)
            step += 1

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

        if current_epoch == int(config.num_epoch):
            break

    print('Training Complete...\n')
    sess.close()


def validation():
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    print('**log_path: ', log_path)
    val_ckpt_path = os.path.join(log_path)
    num_examples = len(dataset.id_test)
    roc_curve_name = '_'.join([config.exp_name, config.npy_name, trial_serial_str, config.model_name,
                               'param-%03d' % (config.num_param)+'_'+config.act_fn]) + '.png'
    result_name = '_'.join([config.exp_name, config.npy_name, trial_serial_str, config.model_name,
                            'param-%03d' % (config.num_param) + '_' + config.act_fn]) + '.csv'

    print('num_examples: ', num_examples)

    saver = tf.train.Saver()  # default graph
    prep_saver = tf.train.Saver(var_list=pretrain_vars)

    ckpt = tf.train.get_checkpoint_state(val_ckpt_path)
    if not ckpt:
        raise ValueError('No checkpoint found in ' + val_ckpt_path)
    all_ckpt_paths = ckpt.all_model_checkpoint_paths
    num_ckpt = len(all_ckpt_paths)

    probs = np.zeros([num_ckpt, num_examples, model.class_num])
    lbls = np.zeros([num_examples], dtype=np.int32)

    for ckpt_idx, ckpt_path in enumerate(all_ckpt_paths):
        print('Restroing: '+ckpt_path)

        sess = tf.Session(config=sess_config)
        #sess.run(init_op)
        saver.restore(sess, ckpt_path)

        sess.run(dataset.test.init_op)

        test_X, test_Y = [], []

        num_iter = int(np.ceil(float(num_examples) / config.batch_size))
        step = 0

        while step < num_iter:
            sys.stdout.write('Evaluation [{0}/{1}]\r'.format(len(test_X) // config.batch_size,
                                                             -(-dataset.test.data_length // config.batch_size)))

            img1, img2, img3, lbl, id, age, sxf, sxm, vas, tm0, tm1, tm2, dm0, dm1, dm2 = \
                sess.run(dataset.test.next_batch)
            imgs = dict({1: img1, 2: img2, 3: img3})

            conv = prep_weight_process(sess, pre_path, prep_saver, imgs, id,
                                       model.image_height, model.image_width, model.class_num)

            feed_dict = {model.images1: conv[0], model.images2: conv[1], model.images3: conv[2],
                         model.labels: lbl, model.ages: age, model.sxf: sxf, model.sxm: sxm,
                         model.vas: vas, model.tm0: tm0, model.tm1: tm1, model.tm2: tm2,
                         model.dm0: dm0, model.dm1: dm1, model.dm2: dm2, model.is_training: False,
                         prev_model.images: np.zeros([len(id), prev_model.image_height, prev_model.image_width, 1]),
                         prev_model.labels: np.zeros([len(id)]), prev_model.is_training: False
                         }

            test_outputs, test_logits = sess.run([model.logits, tf.nn.softmax(model.logits)], feed_dict=feed_dict)

            probs[ckpt_idx, step*config.batch_size:step*config.batch_size+len(id)] = test_logits

            if ckpt_idx == 0:
                lbls[step*config.batch_size:step*config.batch_size+len(id)] = lbl

            test_X.extend(test_logits)
            step += 1

        sess.close()

    probs = np.mean(probs, axis=0)
    id_test = dataset.id_test

    prob_0, prob_1 = np.array(probs[:, 0]), np.array(probs[:, 1])
    result_csv = pd.DataFrame({'NUMBER': id_test, 'PROB_0': prob_0, 'PROB_1': prob_1, 'LABEL': lbls})
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)
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

