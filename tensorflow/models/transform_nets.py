import tensorflow as tf
import numpy as np
import sys
import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '../utils'))
# import tf_util

class TransXYZ(tf.keras.layers.Layer):

  def __init__(self, K=3, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.K = K

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    out = inputs
    with tf.compat.v1.variable_scope('transform_XYZ'):
      # The weights are created with a `regularizer`,
      # so the layer should track their regularization losses
      weights = tf.compat.v1.get_variable('weights', [256, self.K*self.K],
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
      biases = tf.compat.v1.get_variable('biases', [self.K*self.K],
                   initializer=tf.constant_initializer(0.0),
                   dtype=tf.float32)
      biases.assign_add(tf.constant(np.eye(self.K).flatten(), dtype=tf.float32))
      out = tf.matmul(out, weights)
      out = tf.compat.v1.nn.bias_add(out, biases)
    return out


def input_transform_net(edge_feature,K=3):
  """ Input (XYZ) Transform Net, input is BxNx3 gray image
    Return:
      Transformation matrix of size 3xK """
  batch_size = edge_feature.shape[0]
  num_point = edge_feature.shape[1]
  print("Inside transform net")

  
  x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=True, padding='valid')(edge_feature)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.Conv2D(128, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)

  x = tf.keras.layers.Conv2D(1024, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.MaxPooling2D(pool_size=(num_point,1),strides=(2,2))(x)
  
  x = tf.keras.layers.Flatten()(x)
  
  x = tf.keras.layers.Dense(512,activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.Dense(256,activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  transform = tf.keras.layers.Dense(K*K)(x)

  transform = tf.keras.layers.Reshape((K,K))(transform)

  
  return transform
