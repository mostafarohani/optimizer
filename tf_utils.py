import tensorflow as tf

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
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  pass
  # with tf.name_scope('summaries'):
  #   mean = tf.reduce_mean(var)
  #   tf.scalar_summary('mean/' + name, mean)
  # with tf.name_scope('stddev'):
  #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  # tf.scalar_summary('sttdev/' + name, stddev)
  # tf.scalar_summary('max/' + name, tf.reduce_max(var))
  # tf.scalar_summary('min/' + name, tf.reduce_min(var))
  # tf.histogram_summary(name, var)
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
    activations = act(preactivate, 'activation')
    # tf.histogram_summary(layer_name + '/activations', activations)
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
    # tf.histogram_summary(layer_name + '/pre_maxpool', h_conv)
    h_pool = max_pool_2x2(h_conv)
    # tf.histogram_summary(layer_name + '/post_maxpool', h_pool)
    return h_pool