import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers


def vas_prep_missed(vas_value):
    vas_value = tf.squeeze(vas_value)
    missing_value = tf.constant(99, tf.float32)
    t_new = tf.map_fn(
        lambda x: tf.case(
            pred_fn_pairs=[
                (tf.equal(x, missing_value), lambda: tf.constant(6, tf.float32)),
                (tf.not_equal(x, missing_value), lambda: x),
                ],
            default=lambda: tf.constant(-1, tf.float32)),
        vas_value)
    t_new = tf.expand_dims(t_new, 1)
    return t_new


def get_global_vars(scope_list):
    _vars = []
    for scope in scope_list:
        _vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return _vars


def get_prep_conv_vars():
    scope_list = ['model2_2', 'Classifier']
    _vars = get_global_vars(scope_list)
    return _vars


class Aggre_network():
    def __init__(self, num_param, act_fn, trainable=False):
        self.image_height, self.image_width, self.image_channel = [16, 16, 2]

        self.images1 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images2 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images3 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.list_images = [self.images1, self.images2, self.images3]

        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.class_num = 2
        self.is_training = tf.placeholder(tf.bool, shape=None)

        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 1), name='positive'))

        with tf.name_scope('age'):
            self.ages = tf.placeholder('float32', [None, ])
            ages = tf.reshape(self.ages, [-1, 1])

        with tf.name_scope('sex'):
            self.sxf = tf.placeholder('float32', [None, ])
            self.sxm = tf.placeholder('float32', [None, ])
            sxf = tf.reshape(self.sxf, [-1, 1])
            sxm = tf.reshape(self.sxm, [-1, 1])

        with tf.name_scope('vas'):
            self.vas = tf.placeholder('float32', [None, ])
            vas = tf.reshape(self.vas, [-1, 1])

        with tf.name_scope('trauma'):
            self.tm0 = tf.placeholder('float32', [None, ])
            self.tm1 = tf.placeholder('float32', [None, ])
            self.tm2 = tf.placeholder('float32', [None, ])
            tm0 = tf.reshape(self.tm0, [-1, 1])
            tm1 = tf.reshape(self.tm1, [-1, 1])
            tm2 = tf.reshape(self.tm2, [-1, 1])

        with tf.name_scope('dominant'):
            self.dm0 = tf.placeholder('float32', [None, ])
            self.dm1 = tf.placeholder('float32', [None, ])
            self.dm2 = tf.placeholder('float32', [None, ])
            dm0 = tf.reshape(self.dm0, [-1, 1])
            dm1 = tf.reshape(self.dm1, [-1, 1])
            dm2 = tf.reshape(self.dm2, [-1, 1])

        lse = []
        with tf.name_scope('Aggregate'):
            for idx in range(0, 3):
                each_lse = tf.reduce_logsumexp(self.list_images[idx], axis=[1,2], keepdims=False)
                lse.append(each_lse)
        print('lse: ', lse)
        with tf.name_scope('Add_clinical'):
            aggre_lse = tf.concat(lse, axis=1)
            self.add = tf.concat([aggre_lse, ages*0.01, sxf, sxm,
                                  vas*0.1, tm0, tm1, tm2, dm0, dm1, dm2],
                                 axis=1)
        print('add: ', self.add)
        if act_fn == 'sigmoid':
            fc = layers.fully_connected(inputs=self.add, num_outputs=num_param, activation_fn=tf.nn.sigmoid)
        elif act_fn == 'relu':
            fc = layers.fully_connected(inputs=self.add, num_outputs=num_param, activation_fn=tf.nn.relu)
        else:
            raise ValueError('Error! Invalid activation function.')

        print('fc: ', fc)

        self.logits = layers.fully_connected(inputs=fc, num_outputs=self.class_num,
                                             activation_fn=None)
        print('logits: ', self.logits)

        is_correct = tf.equal(tf.argmax(self.logits, axis=1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=self.logits))

        if trainable:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

            # regularization loss
            reg_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=tf.train.get_global_step(),
                                                       decay_steps=5000, decay_rate=0.94, staircase=True)  # 2000

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, epsilon=0.1)
                self.train = optimizer.minimize(self.loss + reg_loss, global_step=tf.train.get_global_step())


class Aggre_network2():
    def __init__(self, num_param, act_fn=None, trainable=False):
        self.image_height, self.image_width, self.image_channel = [16, 16, 2]

        self.images1 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images2 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images3 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.list_images = [self.images1, self.images2, self.images3]

        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.class_num = 2
        self.is_training = tf.placeholder(tf.bool, shape=None)

        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 1), name='positive'))

        with tf.name_scope('age'):
            self.ages = tf.placeholder('float32', [None, ])
            ages = tf.reshape(self.ages, [-1, 1])

        with tf.name_scope('sex'):
            self.sxf = tf.placeholder('float32', [None, ])
            self.sxm = tf.placeholder('float32', [None, ])
            sxf = tf.reshape(self.sxf, [-1, 1])
            sxm = tf.reshape(self.sxm, [-1, 1])

        with tf.name_scope('vas'):
            self.vas = tf.placeholder('float32', [None, ])
            vas = tf.reshape(self.vas, [-1, 1])

        with tf.name_scope('trauma'):
            self.tm0 = tf.placeholder('float32', [None, ])
            self.tm1 = tf.placeholder('float32', [None, ])
            self.tm2 = tf.placeholder('float32', [None, ])
            tm0 = tf.reshape(self.tm0, [-1, 1])
            tm1 = tf.reshape(self.tm1, [-1, 1])
            tm2 = tf.reshape(self.tm2, [-1, 1])

        with tf.name_scope('dominant'):
            self.dm0 = tf.placeholder('float32', [None, ])
            self.dm1 = tf.placeholder('float32', [None, ])
            self.dm2 = tf.placeholder('float32', [None, ])
            dm0 = tf.reshape(self.dm0, [-1, 1])
            dm1 = tf.reshape(self.dm1, [-1, 1])
            dm2 = tf.reshape(self.dm2, [-1, 1])

        lse = []
        with tf.name_scope('Aggregate'):
            for idx in range(0, 3):
                each_lse = tf.reduce_logsumexp(self.list_images[idx], axis=[1,2], keepdims=False)
                lse.append(each_lse)
        print('lse: ', lse)
        with tf.name_scope('Add_clinical'):
            aggre_lse = tf.concat(lse, axis=1)
            self.fc = tf.concat([aggre_lse, ages*0.01, sxf, sxm,
                                  vas*0.1, tm0, tm1, tm2, dm0, dm1, dm2],
                                 axis=1)
        print('add: ', self.fc)
        if act_fn == 'sigmoid':
            for out_param in [16, 16]:
                self.fc = layers.fully_connected(inputs=self.fc, num_outputs=out_param, activation_fn=tf.nn.sigmoid)
                print('fc: ', self.fc)
        elif act_fn == 'relu':
            for out_param in [16, 16]:
                self.fc = layers.fully_connected(inputs=self.fc, num_outputs=out_param, activation_fn=tf.nn.relu)
                print('fc: ', self.fc)
        else:
            raise ValueError('Error! Invalid activation function.')

        #print('fc: ', self.fc)

        self.logits = layers.fully_connected(inputs=self.fc, num_outputs=self.class_num,
                                             activation_fn=None)
        print('logits: ', self.logits)

        is_correct = tf.equal(tf.argmax(self.logits, axis=1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=self.logits))
        if trainable:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

            # regularization loss
            reg_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=tf.train.get_global_step(),
                                                       decay_steps=5000, decay_rate=0.94, staircase=True)  # 2000

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, epsilon=0.1)
                self.train = optimizer.minimize(self.loss + reg_loss, global_step=tf.train.get_global_step())


class Aggre_network3():
    def __init__(self, num_param, act_fn=None, trainable=False):
        self.image_height, self.image_width, self.image_channel = [16, 16, 2]

        self.images1 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images2 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images3 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.list_images = [self.images1, self.images2, self.images3]

        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.class_num = 2
        self.is_training = tf.placeholder(tf.bool, shape=None)

        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 1), name='positive'))

        with tf.name_scope('age'):
            self.ages = tf.placeholder('float32', [None, ])
            ages = tf.reshape(self.ages, [-1, 1])

        with tf.name_scope('sex'):
            self.sxf = tf.placeholder('float32', [None, ])
            self.sxm = tf.placeholder('float32', [None, ])
            sxf = tf.reshape(self.sxf, [-1, 1])
            sxm = tf.reshape(self.sxm, [-1, 1])

        with tf.name_scope('vas'):
            self.vas = tf.placeholder('float32', [None, ])
            vas = tf.reshape(self.vas, [-1, 1])

        with tf.name_scope('trauma'):
            self.tm0 = tf.placeholder('float32', [None, ])
            self.tm1 = tf.placeholder('float32', [None, ])
            self.tm2 = tf.placeholder('float32', [None, ])
            tm0 = tf.reshape(self.tm0, [-1, 1])
            tm1 = tf.reshape(self.tm1, [-1, 1])
            tm2 = tf.reshape(self.tm2, [-1, 1])

        with tf.name_scope('dominant'):
            self.dm0 = tf.placeholder('float32', [None, ])
            self.dm1 = tf.placeholder('float32', [None, ])
            self.dm2 = tf.placeholder('float32', [None, ])
            dm0 = tf.reshape(self.dm0, [-1, 1])
            dm1 = tf.reshape(self.dm1, [-1, 1])
            dm2 = tf.reshape(self.dm2, [-1, 1])

        lse = []
        with tf.name_scope('Aggregate'):
            for idx in range(0, 3):
                each_lse = tf.reduce_logsumexp(self.list_images[idx], axis=[1,2], keepdims=False)
                lse.append(each_lse)
        print('lse: ', lse)

        with tf.name_scope('Add_clinical'):
            self.add = tf.concat(lse, axis=1)

        print('add: ', self.add)
        if act_fn == 'sigmoid':
            fc = layers.fully_connected(inputs=self.add, num_outputs=num_param, activation_fn=tf.nn.sigmoid)
        elif act_fn == 'relu':
            fc = layers.fully_connected(inputs=self.add, num_outputs=num_param, activation_fn=tf.nn.relu)
        else:
            raise ValueError('Error! Invalid activation function.')
        print('fc: ', fc)

        self.logits = layers.fully_connected(inputs=fc, num_outputs=self.class_num, activation_fn=None)
        print('logits: ', self.logits)

        is_correct = tf.equal(tf.argmax(self.logits, axis=1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=self.logits))
        if trainable:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

            # regularization loss
            reg_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=tf.train.get_global_step(),
                                                       decay_steps=5000, decay_rate=0.94, staircase=True)  # 2000

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, epsilon=0.1)
                self.train = optimizer.minimize(self.loss + reg_loss, global_step=tf.train.get_global_step())


class Aggre_logistic():
    def __init__(self, num_param=None, act_fn=None, trainable=False):
        self.image_height, self.image_width, self.image_channel = [16, 16, 2]

        self.images1 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images2 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.images3 = tf.placeholder(tf.float32, shape=[None, self.image_height,
                                                         self.image_width, self.image_channel])
        self.list_images = [self.images1, self.images2, self.images3]

        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.class_num = 2
        self.is_training = tf.placeholder(tf.bool, shape=None)

        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images1, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images2, tf.equal(self.labels, 1), name='positive'))

        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images3, tf.equal(self.labels, 1), name='positive'))

        with tf.name_scope('age'):
            self.ages = tf.placeholder('float32', [None, ])
            ages = tf.reshape(self.ages, [-1, 1])

        with tf.name_scope('sex'):
            self.sxf = tf.placeholder('float32', [None, ])
            self.sxm = tf.placeholder('float32', [None, ])
            sxf = tf.reshape(self.sxf, [-1, 1])
            sxm = tf.reshape(self.sxm, [-1, 1])

        with tf.name_scope('vas'):
            self.vas = tf.placeholder('float32', [None, ])
            vas = tf.reshape(self.vas, [-1, 1])

        with tf.name_scope('trauma'):
            self.tm0 = tf.placeholder('float32', [None, ])
            self.tm1 = tf.placeholder('float32', [None, ])
            self.tm2 = tf.placeholder('float32', [None, ])
            tm0 = tf.reshape(self.tm0, [-1, 1])
            tm1 = tf.reshape(self.tm1, [-1, 1])
            tm2 = tf.reshape(self.tm2, [-1, 1])

        with tf.name_scope('dominant'):
            self.dm0 = tf.placeholder('float32', [None, ])
            self.dm1 = tf.placeholder('float32', [None, ])
            self.dm2 = tf.placeholder('float32', [None, ])
            dm0 = tf.reshape(self.dm0, [-1, 1])
            dm1 = tf.reshape(self.dm1, [-1, 1])
            dm2 = tf.reshape(self.dm2, [-1, 1])

        lse = []
        with tf.name_scope('Aggregate'):
            for idx in range(0, 3):
                each_lse = tf.reduce_logsumexp(self.list_images[idx], axis=[1,2], keepdims=False)
                lse.append(each_lse)
        print('lse: ', lse)

        with tf.name_scope('Add_clinical'):
            aggre_lse = tf.concat(lse, axis=1)
            self.add = tf.concat([aggre_lse, ages*0.01, sxf, sxm,
                                  vas*0.1, tm0, tm1, tm2, dm0, dm1, dm2],
                                 axis=1)

        with tf.name_scope('Logistic'):
            w = tf.get_variable(name='weight', shape=[16, 1], initializer=tf.truncated_normal_initializer())
            b = tf.get_variable(name='bias', shape=[1], initializer=tf.truncated_normal_initializer())
            self.logits = tf.sigmoid(tf.matmul(self.add, w) + b)

        labels = tf.cast(self.labels, tf.float32)

        predicted = tf.cast(self.logits > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels),
                                       dtype=tf.float32))

        self.loss = -tf.reduce_mean(labels*tf.log(self.logits)+(1-labels)*(tf.log(1-self.logits)))

        if trainable:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
                self.train = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())




if __name__ == '__main__':
    #Aggre_network2(num_param=12, act_fn='relu')
    Aggre_logistic()
