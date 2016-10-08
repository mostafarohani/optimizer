# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
from tf_utils import *
from gn  import *

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.009, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/optimizer_simple_logs', 'Summaries directory')

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      # tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, name = 'activation')
    # tf.histogram_summary(layer_name + '/activations', activations)
    return weights, activations
def train():

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                one_hot=True,
                                fake_data=FLAGS.fake_data)
  sess = tf.InteractiveSession()
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
  weights, y_hat = nn_layer(x, 784, 10, 'single_layer',act=tf.nn.softmax)
  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y_hat)
    cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)
  with tf.name_scope('train'):  
    #optimizer  = tf.train.AdamOptimizer(FLAGS.learning_rate)
    print(type(cross_entropy))
    optimizer  = GNOptimizer(cross_entropy, y_hat, learning_rate = FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy)
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
  merged = tf.merge_all_summaries()

  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  gradstep = tf.gradients(y_hat, [weights,x])[0]
  # print(type(gradstep))
  # print(len(gradstep))
  # gradstep = optimizer.compute_gradients(cross_entropy)
  tf.initialize_all_variables().run()
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, grads = sess.run([train_step, gradstep], feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
      print(grads.shape, x.get_shape())
      print ("grads ", np.sum(grads))
      # print ("variable ", weight_loss_grads[1])
      print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
