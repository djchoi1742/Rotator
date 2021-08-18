import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers


class Network():
    def __init__(self, trainable=False, is_prep=False):
        self.image_height, self.image_width, self.image_channel = [512, 512, 1]

        self.images = tf.placeholder(tf.float32, shape=[None, self.image_height, self.image_width, self.image_channel])
        self.labels = tf.placeholder(tf.int64, shape=[None])  # sparse index
        self.class_num = 2  # negative vs. positive

        self.is_training = tf.placeholder(tf.bool, shape=None)

        tf.add_to_collection('images', tf.boolean_mask(self.images, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images, tf.equal(self.labels, 1), name='positive'))

        features_2D = self.images
        with tf.variable_scope('model2_2'):
            for in_channels, out_channels in zip([64, 96, 128, 160, 192], [128, 160, 192, 224, 256]):
                features_2D = self.resnet_block(inputs=features_2D, in_channels=in_channels,
                                                out_channels=out_channels,
                                                use_attention=True, is_training=self.is_training)

                features_2D = layers.max_pool2d(inputs=features_2D, kernel_size=2, stride=2)
                print('features_2D: ', features_2D)

        with tf.variable_scope('Classifier'):  # pixelwise classification (local classification)
            for in_channels, out_channels in zip([224], [self.class_num]):
                self.features_2D = self.resnet_block(inputs=features_2D, in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     use_attention=True, is_training=self.is_training)

            # aggregation of local classification into the global classification result
            self.logits = tf.reduce_logsumexp(self.features_2D, axis=[1, 2], keepdims=False)

            print('logits: ', self.logits)

        is_correct = tf.equal(tf.argmax(self.logits, axis=1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=self.logits))
        self.local_classification = self.features_2D

        self.grad_cam = self.visualize_grad_cam(loss=self.loss, conv=self.features_2D,
                                                image_size=[self.image_height, self.image_width])

        self.local_class = tf.image.resize_bilinear(images=self.features_2D,
                                                    size=[self.image_height, self.image_width])
        self.local_neg = tf.expand_dims(tf.nn.relu(self.local_class[:, :, :, 0]), axis=-1)
        self.local_pos = tf.expand_dims(tf.nn.relu(self.local_class[:, :, :, 1]), axis=-1)
        print('grad_cam: ', self.grad_cam.shape)

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

    def resnet_block(self, inputs, in_channels, out_channels, use_attention, is_training):
        conv = layers.conv2d(inputs=inputs, num_outputs=in_channels, kernel_size=1, stride=1, activation_fn=None)
        conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

        conv = layers.conv2d(inputs=conv, num_outputs=in_channels, kernel_size=3, stride=1, activation_fn=None)
        conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

        conv = layers.conv2d(inputs=conv, num_outputs=out_channels, kernel_size=1, stride=1, activation_fn=None)
        conv = layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training)

        def attention_block(inputs, reduction_ratio=4):
            squeeze = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)  # global average pooling
            excitation = layers.fully_connected(inputs=squeeze, num_outputs=squeeze.shape[-1].value // reduction_ratio,
                                                activation_fn=tf.nn.relu)
            excitation = layers.fully_connected(inputs=excitation, num_outputs=squeeze.shape[-1].value,
                                                activation_fn=tf.nn.sigmoid)
            outputs = inputs * excitation
            return outputs

        if use_attention:
            conv = attention_block(inputs=conv, reduction_ratio=16)

        if not inputs.shape[-1].value == out_channels:
            inputs = layers.conv2d(inputs=inputs, num_outputs=out_channels, kernel_size=1, stride=1, activation_fn=None)
            inputs = layers.batch_norm(inputs=inputs, center=True, scale=True, is_training=is_training)

        if out_channels == self.class_num:
            return conv+inputs
        else:
            return tf.nn.relu(conv+inputs)

    def visualize_grad_cam(self, loss, conv, image_size):
        grads = tf.gradients(loss, conv)[0]  # because here conv argument has a single layer
        # normalizing the gradients
        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        conv_shape = conv.shape

        conv = tf.reshape(conv, [-1, np.prod(conv_shape[1:-1]), conv_shape[-1]])  # N H*W C
        weights = tf.expand_dims(tf.reduce_max(norm_grads, axis=[1, 2]), axis=-1)  # N C 1

        cam = tf.matmul(conv, weights)  # N H*W 1
        cam = tf.reshape(cam, [-1, conv_shape[1], conv_shape[2], 1])
        # zero-out using relu (to obtain positive effect only) and resizing heatmap
        cam = tf.nn.relu(cam)
        cam = tf.image.resize_bilinear(images=cam, size=image_size)

        return cam


if __name__ == '__main__':
    senet = Network()
    print(tf.global_variables())
