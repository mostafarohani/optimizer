import tensorflow as tf
import sys
import numpy as np
from tf_utils import *
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/optimizer_logs', 'Summaries directory')

optimizer = {}

optimizer["adam"] = tf.train.AdamOptimizer
optimizer["adadelta"] = tf.train.AdadeltaOptimizer
optimizer["adagrad"] = tf.train.AdagradOptimizer
optimizer["momentum"] = lambda x: tf.train.MomentumOptimizer(x, 0.9, use_nesterov = True)
optimizer["sgd"] = tf.train.GradientDescentOptimizer
optimizer["rmsprop"] = tf.train.RMSPropOptimizer
# optimizer["ftrl"] = tf.train.FtrlOptimizer


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

  ### DEFINING NEURAL NET STRUCTURE ###
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
      x_image = tf.reshape(x, [-1, 28, 28, 1])
      tf.image_summary('input', x_image, 10)

  conv_1 = conv_max_layer(x_image, (5,5), 1, 32, 'convolution_layer_1')
  conv_2 = conv_max_layer(conv_1, (5,5), 32, 64, 'convolution_layer_2')

  conv2_flat = tf.reshape(conv_2, [-1, 7*7*64])

  h_fc1 = nn_layer(conv2_flat, 7*7*64, 1024, 'fully_connected_layer_1')

  with tf.name_scope('h_fc1_dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.scalar_summary('dropout_keep_probability', keep_prob)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  y_hat = nn_layer(h_fc1_drop, 1024, 10, 'fully_connected_layer_2', act=tf.nn.softmax)

  # defining optimization parameters
  with tf.name_scope('cross_entropy'):
      diff = y_ * tf.log(y_hat)
      with tf.name_scope('total'):
          cross_entropy = -tf.reduce_mean(diff)
      tf.scalar_summary('cross entropy', cross_entropy)
  # with tf.name_scope('train'):
      # if optimizer_name in optimizer:
      # train_step = optimizer[FLAGS.optimizer_name](FLAGS.learning_rate).minimize(cross_entropy)
      # else:
      #   print(optimizer_name + " is not one of the supported optimizers. Using Adam")
      #   train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)


  ### FINISHED DEFINING NEURAL NET STRUCTURE ###

  # Merge all the summaries and write them out to /tmp/optimizer_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  best_accs = []
  for optimizer_name in optimizer.keys():
    best_acc = -1
    best_learning_rate = 0
    for learning_rate in [10**(-1*i) for i in np.arange(1, 4, 0.5)]:
      print (optimizer_name, learning_rate)
      solver = optimizer[optimizer_name](learning_rate)
      train_op = solver.minimize(cross_entropy)  
      tf.initialize_all_variables().run()
      for i in range(FLAGS.max_steps):
        if i % 100 == 0:  # Record summaries and test-set accuracy
          summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
          test_writer.add_summary(summary, i)
          print('Testing Accuracy at step %s: %s' % (i, acc))
          if acc > best_acc:
            best_learning_rate = learning_rate
            best_acc = acc
        else:  # Record train set summaries, and train
          # if i % 100 == 99 and False:  # Record execution stats
          #   run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          #   run_metadata = tf.RunMetadata()
          #   summary, _ = sess.run([merged, train_step],
          #             feed_dict=feed_dict(True),
          #             options=run_options,
          #             run_metadata=run_metadata)
          #   train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          #   train_writer.add_summary(summary, i)
          #   print('Adding run metadata for', i)
          # else:  # Record a summary
          summary, _ = sess.run([merged, train_op], feed_dict=feed_dict(True))
          train_writer.add_summary(summary, i)
    best_accs.append((optimizer_name, best_acc, best_learning_rate))
  train_writer.close()
  test_writer.close()
  sess.close()
  print (best_accs)

def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
