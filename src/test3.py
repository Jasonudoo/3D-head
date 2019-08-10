# -*- coding=utf-8 -*-  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys
import tables
import numpy as np
from tensorflow.python.ops import control_flow_ops
import random
import math
from net import nets_factory
#from auxiliary import losses
#from roc_curve import calculate_roc
from file_tools import different_test
#import os
from file_tools import different_2d_test
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "2"



slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '../model/vgg16_adagrad_summmary-CAS',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(

    'development_dataset_path', '../../data/development_sample_dataset_speaker.hdf5',#data/development_sample_dataset_speaker.hdf5####
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 4,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_boolean('online_pair_selection', False,
                            'Use online pair selection.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 1,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 10,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 500,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')
######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'adagrad',#adam---adadelta
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',#learning_rate
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 5.0,#5.0--20.0
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################


tf.app.flags.DEFINE_string(
    'model_speech', 'cnn_speech', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch. It will be the number of samples distributed for all clones.')#3-30

tf.app.flags.DEFINE_integer(
    'num_epochs', 1, 'The number of epochs for training.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS
def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    print('FLAGS.batch_size=',FLAGS.batch_size,'FLAGS.num_epochs_per_decay=',FLAGS.num_epochs_per_decay)
    print('decay_steps=',decay_steps)
    """decay steps"""
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate
        #print('decay_step',decay_steps)
    #print('FLAGS.learning_rate_decay_type=',FLAGS.learning_rate_decay_type)    

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    print('FLAGS.optimizer=',FLAGS.optimizer)
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            #print('g=',g)
            expanded_g = tf.expand_dims(g, 0)
            #print('expanded_g=',expanded_g)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
            #print('grads=',grads)
            #print('')

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        #print('grad1=',grad)
        grad = tf.reduce_mean(grad, 0)
        #print('grad2=',grad)
        #print('')

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads




num_subjects = 10

def main(_):

    # Log
    tf.logging.set_verbosity(tf.logging.INFO)
    
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        #########################################
        ########## required from data ###########
        #########################################


        
        img, label = different_test.read_and_decode("../data1/tfrecord-test-3/")
        img_test,label_test = img,label
        num_samples_per_epoch = FLAGS.batch_size * 500


        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)
        



        # Create global_step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        #####################################
        #### Configure the larning rate. ####
        #####################################
        learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        opt = _configure_optimizer(learning_rate)

        ######################
        # Select the network #
        ######################

        # Training flag.
        is_training = tf.placeholder(tf.bool)

        # Get the network. The number of subjects is num_subjects.
        model_speech_fn = nets_factory.get_network_fn(
            FLAGS.model_speech,
            num_classes=num_subjects,
            weight_decay=FLAGS.weight_decay,
            is_training=is_training)
    


        #####################################
        # Select the preprocessing function #
        #####################################

        # TODO: Do some preprocessing if necessary.

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        # with tf.device(deploy_config.inputs_device()):
        """
        Define the place holders and creating the batch tensor.
        """
        #img = tf.placeholder(tf.float32, (224,224, 3))
        #label = tf.placeholder(tf.int32, (1))
        batch_dynamic = tf.placeholder(tf.int32, ())
        margin_imp_tensor = tf.placeholder(tf.float32, ())

        img_batch, label_batch = tf.train.shuffle_batch([img_test, label_test],
                                                        batch_size=FLAGS.batch_size, capacity=2000,
                                                        min_after_dequeue=1000)

        
   
        #############################
        # Specify the loss function #
        #############################
        tower_grads = []
        tower_a = []
        trained_label = []
        true_label = []
        total_right=0
        total = 0
        tp_arr=[]
        tn_arr=[]
        a=1
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_clones):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        """
                        Two distance metric are defined:
                           1 - distance_weighted: which is a weighted average of the distance between two structures.
                           2 - distance_l2: which is the regular l2-norm of the two networks outputs.
                        Place holders
                        """

                        ########################################
                        ######## Outputs of two networks #######
                        ########################################

                        # Distribute data among all clones equally.
                        step = int(FLAGS.batch_size / float(FLAGS.num_clones))
                        #STEP=10

                        # Network outputs.
                        
                        
                        logits, end_points_speech = model_speech_fn(img_batch[i * step: (i + 1) * step])

                        ###################################
                        ########## Loss function ##########
                        ###################################
                        # one_hot labeling
                        
                        label_onehot = tf.one_hot(label_batch[i * step : (i + 1) * step], depth=num_subjects, axis=-1)
                        # print(label_onehot)
                       # print(label_onehot)

                        #SOFTMAX = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_onehot)
                        SOFTMAX = tf.nn.softmax(logits=logits)

                        # Define loss
                        with tf.name_scope('loss'):
                            #loss = tf.reduce_mean(SOFTMAX)
                            loss = tf.losses.log_loss(SOFTMAX,label_onehot)

                        # Accuracy
                        with tf.name_scope('accuracy'):
                            # Evaluate the model
                            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_onehot, 1))

                            # Accuracy calculation
                            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                            total_right += tf.count_nonzero(correct_pred)


                        # ##### call the optimizer ######
                        # # TODO: call optimizer object outside of this gpu environment
                        #
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        #accuracy=accuracy
                        
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)
                        trained_label.insert(i, tf.argmax(logits, 1))
                        true_label.insert(i, tf.argmax(label_onehot, 1))

                        predictions = trained_label
                        actuals = true_label

                        zeros_like_actuals = tf.zeros_like(actuals)
                        ones_like_actuals = tf.ones_like(actuals)
                        two_like_actuals = tf.constant(2, dtype=tf.int64, shape=ones_like_actuals.shape)
                        three_like_actuals = tf.constant(3, dtype=tf.int64, shape=ones_like_actuals.shape)
                        four_like_actuals = tf.constant(4, dtype=tf.int64, shape=ones_like_actuals.shape)
                        five_like_actuals = tf.constant(5, dtype=tf.int64, shape=ones_like_actuals.shape)
                        six_like_actuals = tf.constant(6, dtype=tf.int64, shape=ones_like_actuals.shape)
                        seven_like_actuals = tf.constant(7, dtype=tf.int64, shape=ones_like_actuals.shape)
                        eight_like_actuals = tf.constant(8, dtype=tf.int64, shape=ones_like_actuals.shape)

                        zeros_like_predictions = tf.zeros_like(predictions)
                        ones_like_predictions = tf.ones_like(predictions)
                        two_like_predictions = tf.constant(2, dtype=tf.int64, shape=ones_like_predictions.shape)
                        three_like_predictions = tf.constant(3, dtype=tf.int64, shape=ones_like_predictions.shape)
                        four_like_predictions = tf.constant(4, dtype=tf.int64, shape=ones_like_predictions.shape)
                        five_like_predictions = tf.constant(5, dtype=tf.int64, shape=ones_like_predictions.shape)
                        six_like_predictions = tf.constant(6, dtype=tf.int64, shape=ones_like_predictions.shape)
                        seven_like_predictions = tf.constant(7, dtype=tf.int64, shape=ones_like_predictions.shape)
                        eight_like_predictions = tf.constant(8, dtype=tf.int64, shape=ones_like_predictions.shape)

                        #tr为true的缩写  表示真实数组中每个类真实有的个数
                        tr0_op = tf.reduce_sum(tf.cast(tf.equal(actuals, zeros_like_actuals), tf.float32))
                        #tp_arr.insert(i,tp0_op)
                        tr1_op = tf.reduce_sum(tf.cast(tf.equal(actuals, ones_like_actuals), tf.float32))
                        #tp_arr.insert(i+1, tp1_op)
                        tr2_op = tf.reduce_sum(tf.cast(tf.equal(actuals, two_like_actuals), tf.float32))
                        #tp_arr.insert(i+2, tp2_op)
                        tr3_op = tf.reduce_sum(tf.cast(tf.equal(actuals, three_like_actuals), tf.float32))
                        #tp_arr.insert(i+3, tp3_op)
                        tr4_op = tf.reduce_sum(tf.cast(tf.equal(actuals, four_like_actuals), tf.float32))
                        tr5_op = tf.reduce_sum(tf.cast(tf.equal(actuals, five_like_actuals), tf.float32))
                        tr6_op = tf.reduce_sum(tf.cast(tf.equal(actuals, six_like_actuals), tf.float32))
                        tr7_op = tf.reduce_sum(tf.cast(tf.equal(actuals, seven_like_actuals), tf.float32))
                        tr8_op = tf.reduce_sum(tf.cast(tf.equal(actuals, eight_like_actuals), tf.float32))

                        #pr是prediction预测的缩写 代表预测的数组中每个类的真实个数
                        pr0_op = tf.reduce_sum(tf.cast(tf.equal(predictions, zeros_like_predictions), tf.float32))
                        pr1_op = tf.reduce_sum(tf.cast(tf.equal(predictions, ones_like_predictions), tf.float32))
                        pr2_op = tf.reduce_sum(tf.cast(tf.equal(predictions, two_like_predictions), tf.float32))
                        pr3_op = tf.reduce_sum(tf.cast(tf.equal(predictions, three_like_predictions), tf.float32))
                        pr4_op = tf.reduce_sum(tf.cast(tf.equal(predictions, four_like_predictions), tf.float32))
                        pr5_op = tf.reduce_sum(tf.cast(tf.equal(predictions, five_like_predictions), tf.float32))
                        pr6_op = tf.reduce_sum(tf.cast(tf.equal(predictions, six_like_predictions), tf.float32))
                        pr7_op = tf.reduce_sum(tf.cast(tf.equal(predictions, seven_like_predictions), tf.float32))
                        pr8_op = tf.reduce_sum(tf.cast(tf.equal(predictions, eight_like_predictions), tf.float32))

                        #tn 代表预测数组和真实数组中每个类的位置和个数都对了的个数
                        tn0_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, zeros_like_actuals),
                                             tf.equal(predictions, zeros_like_predictions)),
                                    tf.float32))
                        tn1_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, ones_like_actuals),
                                             tf.equal(predictions, ones_like_predictions)),
                                    tf.float32))
                        tn2_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, two_like_actuals),
                                             tf.equal(predictions, two_like_predictions)),
                                    tf.float32))
                        tn3_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, three_like_actuals),
                                             tf.equal(predictions, three_like_predictions)),
                                    tf.float32))
                        tn4_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, four_like_actuals),
                                             tf.equal(predictions, four_like_predictions)),
                                    tf.float32))
                        tn5_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, five_like_actuals),
                                             tf.equal(predictions, five_like_predictions)),
                                    tf.float32))
                        tn6_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, six_like_actuals),
                                             tf.equal(predictions, six_like_predictions)),
                                    tf.float32))
                        tn7_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, seven_like_actuals),
                                             tf.equal(predictions, seven_like_predictions)),
                                    tf.float32))
                        tn8_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.equal(actuals, eight_like_actuals),
                                             tf.equal(predictions, eight_like_predictions)),
                                    tf.float32))

                        ac0_op=tf.reduce_sum(tf.cast(tf.equal(actuals, zeros_like_actuals),tf.float32))
                        ac1_op = tf.reduce_sum(tf.cast(tf.equal(actuals, ones_like_actuals), tf.float32))
                        ac2_op = tf.reduce_sum(tf.cast(tf.equal(actuals, two_like_actuals), tf.float32))
                        ac3_op = tf.reduce_sum(tf.cast(tf.equal(actuals, three_like_actuals), tf.float32))
                        ac4_op = tf.reduce_sum(tf.cast(tf.equal(actuals, four_like_actuals), tf.float32))
                        ac5_op = tf.reduce_sum(tf.cast(tf.equal(actuals, five_like_actuals), tf.float32))
                        ac6_op = tf.reduce_sum(tf.cast(tf.equal(actuals, six_like_actuals), tf.float32))
                        ac7_op = tf.reduce_sum(tf.cast(tf.equal(actuals, seven_like_actuals), tf.float32))
                        ac8_op = tf.reduce_sum(tf.cast(tf.equal(actuals, eight_like_actuals), tf.float32))

                        pred0_op = tf.reduce_sum(tf.cast(tf.equal(predictions, zeros_like_predictions), tf.float32))
                        pred1_op = tf.reduce_sum(tf.cast(tf.equal(predictions, ones_like_predictions), tf.float32))
                        pred2_op = tf.reduce_sum(tf.cast(tf.equal(predictions, two_like_predictions), tf.float32))
                        pred3_op = tf.reduce_sum(tf.cast(tf.equal(predictions, three_like_predictions), tf.float32))
                        pred4_op = tf.reduce_sum(tf.cast(tf.equal(predictions, four_like_predictions), tf.float32))
                        pred5_op = tf.reduce_sum(tf.cast(tf.equal(predictions, five_like_predictions), tf.float32))
                        pred6_op = tf.reduce_sum(tf.cast(tf.equal(predictions, six_like_predictions), tf.float32))
                        pred7_op = tf.reduce_sum(tf.cast(tf.equal(predictions, seven_like_predictions), tf.float32))
                        pred8_op = tf.reduce_sum(tf.cast(tf.equal(predictions, eight_like_predictions), tf.float32))


                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        tower_a.append(loss)
        #trained_label = tf.reshape(trained_label, [4, FLAGS.batch_size])
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.

        accuracy_true = tf.reduce_mean(tf.cast(tf.equal(trained_label, true_label), tf.float32))
        grads = average_gradients(tower_grads)
        losses = tf.reduce_mean(tower_a)


        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        #apply_gradient1_op = opt.apply_gradients(accuracys, global_step=global_step)

        # Track the moving averages of all trainable variables.
        MOVING_AVERAGE_DECAY = 0.9999
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)



            # ###################################################
            # ############## TEST PER EACH EPOCH ################
            # ###################################################

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        # Initialization of the network.
        variables_to_restore = slim.get_variables_to_restore()

        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, FLAGS.train_dir)


        #####################################
        ############## TRAIN ################
        #####################################

        tower_accuracy=[]
        tower_accuracy_true = []
        tower_losses = []
        tower_recall = []
        tower_precision = []
        tower_F1 = []
        tower_recall0 = []
        tower_recall1 = []
        tower_recall2 = []
        tower_recall3 = []
        tower_recall4 = []
        tower_recall5 = []
        tower_recall6 = []
        tower_recall7 = []
        tower_recall8 = []

        tower_precision0 = []
        tower_precision1= []
        tower_precision2 = []
        tower_precision3 = []
        tower_precision4 = []
        tower_precision5 = []
        tower_precision6 = []
        tower_precision7 = []
        tower_precision8 = []

        tower_F10 = []
        tower_F11 = []
        tower_F12 = []
        tower_F13 = []
        tower_F14 = []
        tower_F15 = []
        tower_F16 = []
        tower_F17 = []
        tower_F18 = []

        threads = tf.train.start_queue_runners(sess=sess)
        for epoch in range(FLAGS.num_epochs):


            step=0
            a0 = 0
            a1 = 0
            a2 = 0
            a3 = 0
            a4 = 0
            a5 = 0
            a6 = 0
            a7 = 0
            a8 = 0

            b0 = 0
            b1 = 0
            b2 = 0
            b3 = 0
            b4 = 0
            b5 = 0
            b6 = 0
            b7 = 0
            b8 = 0

            c0 = 0
            c1 = 0
            c2 = 0
            c3 = 1
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0
            c8 = 0
            for batch_num in range(num_batches_per_epoch):
                step += 1

                tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8,pr0,pr1,pr2,pr3,pr4,pr5,pr6,pr7,pr8,t0,t1,t2,t3,t4,t5,t6,t7,t8,\
                ac0,ac1,ac2,ac3,ac4,ac5,ac6,ac7,ac8,pred0,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,\
                _,label,logits_value,label_value,true_label_value,loss_value, accuracy_value,accuracy_true_value,total_right_value,training_step,trained_label_value,_= sess.run(
                    [tr0_op,tr1_op,tr2_op,tr3_op,tr4_op,tr5_op,tr6_op,tr7_op,tr8_op,
                    pr0_op,pr1_op,pr2_op,pr3_op,pr4_op,pr5_op,pr6_op,pr7_op,pr8_op,
                     tn0_op,tn1_op,tn2_op,tn3_op,tn4_op,tn5_op,tn6_op,tn7_op,tn8_op,
                     ac0_op,ac1_op,ac2_op,ac3_op,ac4_op,ac5_op,ac6_op,ac7_op,ac8_op,
                     pred0_op,pred1_op,pred2_op,pred3_op,pred4_op,pred5_op,pred6_op,pred7_op,pred8_op,
                     img_batch,label_onehot,logits,label_batch,true_label,losses,accuracy,accuracy_true, total_right,global_step,trained_label,is_training],
                    feed_dict={is_training: False})
                print(step)
                print("before training labels:"+str(true_label_value))
                print("after training labels:"+str(trained_label_value))

                '''print(t0)
                print(t1)
                print(t2)
                print(t3)
                print(t4)
                print(t5)
                print(t6)
                print(t7)
                print(t8)
                print(tn0)
                print(tn1)
                print(tn2)
                print(tn3)
                print(tn4)
                print(tn5)
                print(tn6)
                print(tn7)
                print(tn8)'''
                num=0
                num1 = 0


                recallm=0
                precisionm=0

                recalln0 = 0
                recalln1 = 0
                recalln2 = 0
                recalln3 = 0
                recalln4 = 0
                recalln5 = 0
                recalln6 = 0
                recalln7 = 0
                recalln8 = 0

                precisionn0 = 0
                precisionn1 = 0
                precisionn2 = 0
                precisionn3 = 0
                precisionn4 = 0
                precisionn5 = 0
                precisionn6 = 0
                precisionn7 = 0
                precisionn8 = 0


                F10n = 0
                F11n = 0
                F12n = 0
                F13n = 0
                F14n = 0
                F15n = 0
                F16n = 0
                F17n = 0
                F18n = 0




                if tr0!=0:
                    if ac0<pred0:
                        recall0=tr0  / tr0
                    else :
                        recall0 = math.fabs(tr0 + t0 - FLAGS.batch_size) / tr0
                    recalln0 += recall0

                    num += 1
                    if recall0 != 0:
                        recallm += recall0
                        a0 += 1
                    print("0",recall0)

                if pr0!=0:
                    if ac0<pred0:
                        precision0=tr0  / pr0
                    else :
                        precision0=math.fabs(tr0+t0-FLAGS.batch_size)/pr0
                    precisionn0+=precision0

                    num1 += 1
                    if precision0 != 0:
                        precisionm+=precision0
                        b0 += 1

                if tr1!=0 :
                    if ac1<pred1:
                        recall1=tr1  / tr1
                    else:
                        recall1 = math.fabs(tr1+t1 - FLAGS.batch_size) / tr1
                    recalln1 += recall1

                    num += 1
                    if recall1 != 0:
                        recallm += recall1
                        a1 += 1
                    print("1",recall1)

                if pr1!=0:
                    if ac1 < pred1:
                        precision1 = tr1 / pr1
                    else:
                        precision1 = math.fabs(tr1+t1 - FLAGS.batch_size) / pr1
                    precisionn1 += precision1

                    num1+=1
                    if precision1 != 0:
                        precisionm += precision1
                        b1 += 1

                if tr2!=0 :
                    if ac2 < pred2:
                        recall2 = tr2 / tr2
                    else:
                        recall2 = math.fabs(tr2+t2 - FLAGS.batch_size) / tr2
                    recalln2 += recall2

                    num += 1
                    if recall2 != 0:
                        recallm += recall2
                        a2 += 1
                    print("2",recall2)


                if pr2!=0:
                    if ac2 < pred2:
                        precision2 = tr2 / pr2
                    else:
                        precision2 = math.fabs(tr2+t2 - FLAGS.batch_size) / pr2
                    precisionn2 += precision2
                    if precision2 != 0:
                        precisionm += precision2
                        b2+=1
                    num1+=1

                if tr3!=0 :
                    if ac3 < pred3:
                        recall3 = tr3 / tr3
                    else:
                        recall3 = math.fabs(tr3+t3 - FLAGS.batch_size) / tr3
                    recalln3 += recall3

                    num += 1
                    if recall3 != 0:
                        recallm += recall3
                        a3 += 1
                    print("3",recall3)

                if pr3!=0:
                    if ac3 < pred3:
                        precision3 = tr3 / pr3
                    else:
                        precision3 = math.fabs(tr3+t3 - FLAGS.batch_size) / pr3
                    precisionn3 += precision3
                    if precision3 != 0:
                        precisionm += precision3
                        b3+=1
                    num1+=1

                if tr4!=0 :
                    if ac4 < pred4:
                        recall4 = tr4 / tr4
                    else:
                        recall4 = math.fabs(tr4+t4 - FLAGS.batch_size) / tr4
                    recalln4 += recall4

                    num += 1
                    if recall4 != 0:
                        recallm += recall4
                        a4 += 1
                    print("4",recall4)

                if pr4!=0:
                    if ac4 < pred4:
                        precision4 = tr4 / pr4
                    else:
                        precision4 = math.fabs(tr4+t4 - FLAGS.batch_size) / pr4
                    precisionn4 += precision4

                    num1+=1
                    if precision4 != 0:
                        precisionm += precision4
                        b4 += 1

                if tr5!=0 :
                    if ac5 < pred5:
                        recall5 = tr5 / tr5
                    else:
                        recall5 = math.fabs(tr5+t5 - FLAGS.batch_size) / tr5
                    recalln5 += recall5

                    num += 1
                    if recall5 != 0:
                        recallm += recall5
                        a5 += 1
                    print("5",recall5)

                if pr5!=0:
                    if ac5 < pred5:
                        precision5 = tr5 / pr5
                    else:
                        precision5 = math.fabs(tr5+t5 - FLAGS.batch_size) / pr5
                    precisionn5 += precision5

                    num1+=1
                    if precision5 != 0:
                        precisionm += precision5
                        b5 += 1

                if tr6!=0 :
                    if ac6 < pred6:
                        recall6 = tr6 / tr6
                    else:
                        recall6 = math.fabs(tr6+t6 - FLAGS.batch_size) / tr6
                    recalln6 += recall6

                    num += 1
                    if recall6 != 0:
                        recallm += recall6
                        a6 += 1
                    print("6",recall6)

                if pr6!=0:
                    if ac6 < pred6:
                        precision6 = tr6 / pr6
                    else:
                        precision6 = math.fabs(tr6+t6 - FLAGS.batch_size) / pr6
                    precisionn6 += precision6

                    num1+=1
                    if precision6 != 0:
                        precisionm+= precision6
                        b6 += 1

                if tr7!=0 :
                    if ac7 < pred7:
                        recall7 = tr7 / tr7
                    else:
                        recall7 = math.fabs(tr7+t7 - FLAGS.batch_size) / tr7
                    recalln7 += recall7

                    num += 1
                    if recall7 != 0:
                        recallm += recall7
                        a7 += 1
                    print("7",recall7)


                if pr7!=0:
                    if ac7 < pred7:
                        precision7 = tr7 / pr7
                    else:
                        precision7 = math.fabs(tr7+t7 - FLAGS.batch_size) / pr7
                    precisionn7 += precision7
                    if precision7 != 0:
                        precisionm += precision7
                        b7+=1
                    num1+=1

                if tr8!=0 :
                    if ac8 < pred8:
                        recall8 = tr8 / tr8
                    else:
                        recall8 = math.fabs(tr8+t8 - FLAGS.batch_size) / tr8
                    recalln8 += recall8

                    num += 1
                    if recall8 != 0:
                        recallm += recall8
                        a8 += 1
                    print("8",recall8)

                if pr8!=0:
                    if ac8 < pred8:
                        precision8 = tr8 / pr8
                    else:
                        precision8 = math.fabs(tr8+t8 - FLAGS.batch_size) / pr8
                    precisionn8 += precision8

                    num1+=1
                    if precision8 != 0:
                        precisionm += precision8
                        b8 += 1


                '''re0 = recalln0 / a0
                re1 = recalln1 / a1
                re2 = recalln2 / a2
                re3 = recalln3 / a3
                re4 = recalln4 / a4
                re5 = recalln5 / a5
                re6 = recalln6 / a6
                re7 = recalln7 / a7
                re8 = recalln8 / a8

                pre0 = precisionn0 / a0
                pre1 = precisionn1 / a1
                pre2 = precisionn2 / a2
                pre3 = precisionn3 / a3
                pre4 = precisionn4 / a4
                pre5 = precisionn5 / a5
                pre6 = precisionn6 / a6
                pre7 = precisionn7 / a7
                pre8 = precisionn8 / a8'''

                '''if t0!=0 and pr0!=0:
                    recall0=math.fabs(tr0+t0-FLAGS.batch_size)/t0
                    precision0=math.fabs(tr0+t0-FLAGS.batch_size)/pr0
                    recalln0+=recall0
                    precisionn0+=precision0
                    a0+=1
                    num+=1
                    recallm+=recall0
                    precisionm+=precision0
                    #print(recall0)
                if t1!=0 and pr1!=0:
                    recall1=math.fabs(tr1+t1-FLAGS.batch_size)/t1
                    precision1 = math.fabs(tr1 + t1 - FLAGS.batch_size) / pr1
                    recalln1 += recall1
                    precisionn1 += precision1
                    a1 += 1
                    num += 1
                    recallm += recall1
                    precisionm += precision1
                    #print(recall1)
                if t2!=0 and pr2!=0:
                    recall2=math.fabs(tr2+t2-FLAGS.batch_size)/t2
                    precision2 = math.fabs(tr2 + t2 - FLAGS.batch_size) / pr2
                    recalln2 += recall2
                    precisionn2 += precision2
                    a2 += 1
                    num += 1
                    recallm += recall2
                    precisionm += precision2
                    #print(recall2)
                if t3!=0 and pr3!=0:
                    recall3=math.fabs(tr3+t3-FLAGS.batch_size)/t3
                    precision3 = math.fabs(tr3 + t3 - FLAGS.batch_size) / pr3
                    recalln3 += recall3
                    precisionn3 += precision3
                    a3 += 1
                    num += 1
                    recallm += recall3
                    precisionm += precision3
                    #print(recall3)
                if t4!=0 and pr4!=0:
                    recall4=math.fabs(tr4+t4-FLAGS.batch_size)/t4
                    precision4 = math.fabs(tr4 + t4 - FLAGS.batch_size) / pr4
                    recalln4 += recall4
                    precisionn4 += precision4
                    a4 += 1
                    num+=1
                    recallm += recall4
                    precisionm += precision4
                    #print(recall4)
                if t5!=0 and pr5!=0:
                    recall5=math.fabs(tr5+t5-FLAGS.batch_size)/t5
                    precision5 = math.fabs(tr5 + t5 - FLAGS.batch_size) / pr5
                    recalln5 += recall5
                    precisionn5 += precision5
                    a5 += 1
                    num += 1
                    recallm += recall5
                    precisionm += precision5
                    #print(recall5)
                if t6!=0 and pr6!=0:
                    recall6=math.fabs(tr6+t6-FLAGS.batch_size)/t6
                    precision6 = math.fabs(tr6 + t6 - FLAGS.batch_size) / pr6
                    recalln6 += recall6
                    precisionn6 += precision6
                    a6 += 1
                    num += 1
                    recallm += recall6
                    precisionm+= precision6
                    #print(recall6)
                if t7!=0 and pr7!=0:
                    recall7=math.fabs(tr7+t7-FLAGS.batch_size)/t7
                    precision7 = math.fabs(tr7 + t7 - FLAGS.batch_size) / pr7
                    recalln7 += recall7
                    precisionn7 += precision7
                    a7 += 1
                    num += 1
                    recallm += recall7
                    precisionm += precision7
                    #print(recall7)
                if t8!=0 and pr8!=0:
                    recall8=math.fabs(tr8+t8-FLAGS.batch_size)/t8
                    precision8 = math.fabs(tr8 + t8 - FLAGS.batch_size) / pr8
                    recalln8 += recall8
                    precisionn8 += precision8
                    a8 += 1
                    num += 1
                    recallm += recall8
                    precisionm += precision8
                    #print(recall8)'''

                if recalln0 + precisionn0 != 0:
                    F10 = (2 * recalln0 * precisionn0) / (recalln0 + precisionn0)
                    c0+=1
                    F10n+=F10
                    tower_F10.append(F10)


                if  recalln1 + precisionn1 != 0:
                    F11 = (2 * recalln1 * precisionn1) / (recalln1 + precisionn1)
                    c1 += 1
                    F11n += F11
                    tower_F11.append(F11)
                if recalln2 + precisionn2 != 0:
                    F12 = (2 * recalln2 * precisionn2) / (recalln2 + precisionn2)
                    c2 += 1
                    F12n += F12
                    tower_F12.append(F12)
                if recalln3 + precisionn3 != 0:
                    F13 = (2 * recalln3 * precisionn3) / (recalln3 + precisionn3)
                    c3 += 1
                    F13n += F13
                    tower_F13.append(F13)
                if recalln4 + precisionn4 != 0:
                    F14 = (2 * recalln4 * precisionn4) / (recalln4 + precisionn4)
                    c4 += 1
                    F14n += F14
                    tower_F14.append(F14)
                if recalln5 + precisionn5 != 0:
                    F15 = (2 * recalln5 * precisionn5) / (recalln5 + precisionn5)
                    c5 += 1
                    F15n += F15
                    tower_F15.append(F15)
                if recalln6 + precisionn6 != 0:
                    F16 = (2 * recalln6 * precisionn6) / (recalln6 + precisionn6)
                    c6 += 1
                    F16n += F16
                    tower_F16.append(F16)
                if recalln7 + precisionn7 != 0:
                    F17 = (2 * recalln7 * precisionn7) / (recalln7 + precisionn7)
                    c7 += 1
                    F17n += F17
                    tower_F17.append(F17)
                if recalln8 + precisionn8 != 0:
                    F18 = (2 * recalln8 * precisionn8) / (recalln8 + precisionn8)
                    c8 += 1
                    F18n += F18
                    tower_F18.append(F18)


                '''print("recall0:", re0)
                print("recall1:", re1)
                print("recall2:", re2)
                print("recall3:", re3)
                print("recall4:", re4)
                print("recall5:", re5)
                print("recall6:", re6)
                print("recall7:", re7)
                print("recall8:", re8)

                print("precision0:", pre0)
                print("precision1:", pre1)
                print("precision2:", pre2)
                print("precision3:", pre3)
                print("precision4:", pre4)
                print("precision5:", pre5)
                print("precision6:", pre6)
                print("precision7:", pre7)
                print("precision8:", pre8)

                print("F10:", F10)
                print("F11:", F11)
                print("F12:", F12)
                print("F13:", F13)
                print("F14:", F14)
                print("F15:", F15)
                print("F16:", F16)
                print("F17:", F17)
                print("F18:", F18)'''



                recall = recallm / num
                precision = precisionm / num1
                F1 = (2 * recall * precision) / (recall + precision)
                # print(m)

                print("recall:", recall)
                print("precision:", precision)
                print("F1:", F1)
                print("accuracy_true:", accuracy_true_value)
                print("accuracy:",accuracy_value)
                print("loss_value:" + str(loss_value))
                tower_accuracy.append(accuracy_value)
                tower_losses.append(loss_value)
                tower_recall.append(recall)
                tower_precision.append(precision)
                tower_F1.append(F1)
                tower_accuracy_true.append(accuracy_true_value)

                tower_recall0.append(recalln0)
                tower_recall1.append(recalln1)
                tower_recall2.append(recalln2)
                tower_recall3.append(recalln3)
                tower_recall4.append(recalln4)
                tower_recall5.append(recalln5)
                tower_recall6.append(recalln6)
                tower_recall7.append(recalln7)
                tower_recall8.append(recalln8)

                tower_precision0.append(precisionn0)
                tower_precision1.append(precisionn1)
                tower_precision2.append(precisionn2)
                tower_precision3.append(precisionn3)
                tower_precision4.append(precisionn4)
                tower_precision5.append(precisionn5)
                tower_precision6.append(precisionn6)
                tower_precision7.append(precisionn7)
                tower_precision8.append(precisionn8)


                '''tower_F10.append(F10)
                tower_F11.append(F11)
                tower_F12.append(F12)
                tower_F13.append(F13)
                tower_F14.append(F14)
                tower_F15.append(F15)
                tower_F16.append(F16)
                tower_F17.append(F17)
                tower_F18.append(F18)'''


            average_accuracy=sum(tower_accuracy)/num_batches_per_epoch
            average_loss = sum(tower_losses) / num_batches_per_epoch
            average_recall=sum(tower_recall) / num_batches_per_epoch
            average_precision = sum(tower_precision) / num_batches_per_epoch
            average_F1 = sum(tower_F1) / num_batches_per_epoch
            average_accuracy_true = sum(tower_accuracy_true) / num_batches_per_epoch

            '''average_recall0 = sum(tower_recall0) / num_batches_per_epoch
            average_recall1 = sum(tower_recall1) / num_batches_per_epoch
            average_recall2 = sum(tower_recall2) / num_batches_per_epoch
            average_recall3 = sum(tower_recall3) / num_batches_per_epoch
            average_recall4 = sum(tower_recall4) / num_batches_per_epoch
            average_recall5 = sum(tower_recall5) / num_batches_per_epoch
            average_recall6 = sum(tower_recall6) / num_batches_per_epoch
            average_recall7 = sum(tower_recall7) / num_batches_per_epoch
            average_recall8 = sum(tower_recall8) / num_batches_per_epoch

            average_precision0 = sum(tower_precision0) / num_batches_per_epoch
            average_precision1 = sum(tower_precision1) / num_batches_per_epoch
            average_precision2 = sum(tower_precision2) / num_batches_per_epoch
            average_precision3 = sum(tower_precision3) / num_batches_per_epoch
            average_precision4 = sum(tower_precision4) / num_batches_per_epoch
            average_precision5 = sum(tower_precision5) / num_batches_per_epoch
            average_precision6 = sum(tower_precision6) / num_batches_per_epoch
            average_precision7 = sum(tower_precision7) / num_batches_per_epoch
            average_precision8 = sum(tower_precision8) / num_batches_per_epoch

            average_F10 = sum(tower_F10) / num_batches_per_epoch
            average_F11 = sum(tower_F11) / num_batches_per_epoch
            average_F12 = sum(tower_F12) / num_batches_per_epoch
            average_F13 = sum(tower_F13) / num_batches_per_epoch
            average_F14 = sum(tower_F14) / num_batches_per_epoch
            average_F15 = sum(tower_F15) / num_batches_per_epoch
            average_F16 = sum(tower_F16) / num_batches_per_epoch
            average_F17 = sum(tower_F17) / num_batches_per_epoch
            average_F18 = sum(tower_F18) / num_batches_per_epoch'''

            average_recall0 = sum(tower_recall0) / a0
            average_recall1 = sum(tower_recall1) / a1
            average_recall2 = sum(tower_recall2) / a2
            average_recall3 = sum(tower_recall3) / a3
            average_recall4 = sum(tower_recall4) / a4
            average_recall5 = sum(tower_recall5) / a5
            average_recall6 = sum(tower_recall6) / a6
            average_recall7 = sum(tower_recall7) / a7
            average_recall8 = sum(tower_recall8) / a8

            average_precision0 = sum(tower_precision0) / b0
            average_precision1 = sum(tower_precision1) / b1
            average_precision2 = sum(tower_precision2) / b2
            print("tower_precision3：",tower_precision3)
            print(b3)
            average_precision3 = sum(tower_precision3) / b3
            average_precision4 = sum(tower_precision4) / b4
            average_precision5 = sum(tower_precision5) / b5
            average_precision6 = sum(tower_precision6) / b6
            average_precision7 = sum(tower_precision7) / b7
            print("tower_precision7：", tower_precision7)
            print(b7)
            average_precision8 = sum(tower_precision8) / b8

            average_F10 = sum(tower_F10) / c0
            #print("F10：",sum(tower_F10))
            average_F11 = sum(tower_F11) / c1
            average_F12 = sum(tower_F12) / c2
            average_F13 = sum(tower_F13) / c3
            average_F14 = sum(tower_F14) / c4
            average_F15 = sum(tower_F15) / c5
            average_F16 = sum(tower_F16) / c6
            average_F17 = sum(tower_F17) / c7
            average_F18 = sum(tower_F18) / c8




            print("average_accuracy:",average_accuracy)
            print("average_accuracy_true:", average_accuracy_true)
            print("loss_value_average:",average_loss )
            print("average_recall:",average_recall)
            print("average_precision:", average_precision)
            print("average_F1:", average_F1)

            print("average_recall0:", average_recall0)
            print("average_recall1:", average_recall1)
            print("average_recall2:", average_recall2)
            print("average_recall3:", average_recall3)
            print("average_recall4:", average_recall4)
            print("average_recall5:", average_recall5)
            print("average_recall6:", average_recall6)
            print("average_recall7:", average_recall7)
            print("average_recall8:", average_recall8)

            print("average_precision0:", average_precision0)
            print("average_precision1:", average_precision1)
            print("average_precision2:", average_precision2)
            print("average_precision3:", average_precision3)
            print("average_precision4:", average_precision4)
            print("average_precision5:", average_precision5)
            print("average_precision6:", average_precision6)
            print("average_precision7:", average_precision7)
            print("average_precision8:", average_precision8)

            print("average_F10:", average_F10)
            print("average_F11:", average_F11)
            print("average_F12:", average_F12)
            print("average_F13:", average_F13)
            print("average_F14:", average_F14)
            print("average_F15:", average_F15)
            print("average_F16:", average_F16)
            print("average_F17:", average_F17)
            print("average_F18:", average_F18)





if __name__ == '__main__':
    tf.app.run()