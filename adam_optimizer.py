import tensorflow as tf

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


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
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


  def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)
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
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
      activations = act(preactivate, 'activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      return activations
  def conv_max_layer(input_tensor, conv_dim, input_dim, output_dim, layer_name, act =tf.nn.relu):
    with tf.name_scope(layer_name):
      with tf.name_scope('conv_weights'):
        W_conv = weight_variable([conv_dim[0], conv_dim[1], input_dim, output_dim])
        variable_summaries(W_conv, layer_name + '/conv_weights')
      with tf.name_scope('biases'):
        b_conv = bias_variable([output_dim])
        variable_summaries(b_conv, layer_name + '/biases')
      h_conv = act(conv2d(input_tensor, W_conv) + b_conv)
      tf.histogram_summary(layer_name + '/pre_maxpool', h_conv)
      h_pool = max_pool_2x2(h_conv)
      tf.histogram_summary(layer_name + '/post_maxpool', h_pool)
      return h_pool

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
  with tf.name_scope('train'):
      train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
  with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.scalar_summary('accuracy', accuracy)


  ### FINISHED DEFINING NEURAL NET STRUCTURE ###

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  tf.initialize_all_variables().run()

  for i in range(FLAGS.max_steps):
    if i % 100 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Testing Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                  feed_dict=feed_dict(True),
                  options=run_options,
                  run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()
  sess.close()

def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
