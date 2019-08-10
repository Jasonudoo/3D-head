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
from net import nets_factory
#from auxiliary import losses
#from roc_curve import calculate_roc
from file_tools import different_test



slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '../test-model/3dvgg16',
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
    'optimizer', 'adam',#adam---adadelta
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


        
        img, label = different_test.read_and_decode("../data/tfrecord_test/")
        img_test,label_test = img,label
        num_samples_per_epoch = FLAGS.batch_size * 3000


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
                                                        batch_size=30, capacity=2000,
                                                        min_after_dequeue=1000)

        
   
        #############################
        # Specify the loss function #
        #############################
        tower_grads = []
        tower_a = []
        trained_label = []
        total_right=0
        total = 0
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
                        print('img',img_batch) 
                        print('logits',logits)
                        print('a',(a+1))  
                        ###################################
                        ########## Loss function ##########
                        ###################################
                        # one_hot labeling
                        
                        label_onehot = tf.one_hot(label_batch[i * step : (i + 1) * step], depth=num_subjects, axis=-1)
            #            print(label_onehot)
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

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        tower_a.append(loss)
        #trained_label = tf.reshape(trained_label, [4, FLAGS.batch_size])
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.

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
        tower_losses = []
        threads = tf.train.start_queue_runners(sess=sess)
        for epoch in range(FLAGS.num_epochs):


            step=0
            for batch_num in range(num_batches_per_epoch):
                step += 1



                _,label,logits_value,label_value,loss_value, accuracy_value,total_right_value,training_step,trained_label_value,_= sess.run(
                    [img_batch,label_onehot,logits,label_batch,losses,accuracy, total_right,global_step,trained_label,is_training],
                    feed_dict={is_training: False})
                print(step)
                print("before training labels:"+str(label_value))
                print("after training labels:"+str(trained_label_value))
                #print('softmax',sofotmax)


                #print("loss_value:",loss_value)
                #print("accuracy:",accuracy_value)  
                print("accuracy:",accuracy_value)
                print("loss_value:" + str(loss_value))
                tower_accuracy.append(accuracy_value)
                tower_losses.append(loss_value)
            average_accuracy=sum(tower_accuracy)/num_batches_per_epoch
            average_loss = sum(tower_losses) / num_batches_per_epoch
            print("average_accuracy:",average_accuracy)
            print("loss_value_average:"+str(average_loss ))


if __name__ == '__main__':
    tf.app.run()