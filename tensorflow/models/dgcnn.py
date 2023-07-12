import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net
#%%

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl
#%%
class EdgeComp(tf.keras.layers.Layer):

  def __init__(self, k=2):
      super(EdgeComp, self).__init__()
      self.k = k

  def build(self, input_shape):  # Create the state of the layer (weights)
      print("Edge Computing input shape",input_shape)
      
  def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        # input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.k)
  def call(self, inputs):  # Defines the computation from inputs to outputs
      # adj_matrix = tf_util.pairwise_distance(inputs)
      known_axes = [i for i, size in enumerate(inputs.shape) if size == 1]
      if len(known_axes) != 0:
          point_cloud = tf.squeeze(inputs,axis=known_axes)
      else:
          point_cloud = inputs
      point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
      point_cloud_inner = tf.linalg.matmul(point_cloud, point_cloud_transpose)
      point_cloud_inner = -2*point_cloud_inner
      point_cloud_square = tf.math.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True)
      point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
      adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
  
      # nn_idx = tf_util.knn(adj_matrix, k=self.k)
      neg_adj = -adj_matrix
      _, nn_idx = tf.math.top_k(neg_adj, k=self.k)
      
      # edge_feature = tf_util.get_edge_feature(inputs, nn_idx=nn_idx, k=self.k)
      point_cloud_central = point_cloud

      point_cloud_shape = tf.shape(point_cloud)
      batch_size = point_cloud_shape[0]
      num_points = point_cloud_shape[1]
      num_dims = point_cloud_shape[2]

      idx_ = tf.math.multiply(tf.range(batch_size),num_points)
      idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

      point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
      point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
      point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

      point_cloud_central = tf.tile(point_cloud_central, [1, 1, self.k, 1])

      outputs = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
      return outputs
  
#%%
class MatMult(tf.keras.layers.Layer):

  def __init__(self):
      super(MatMult, self).__init__()
      #self.input1 = tf.cast(input1, dtype=tf.float32)
  def build(self, input_shape):  # Create the state of the layer (weights)
      print("Matmult input shape",input_shape)

  def call(self,inputs):  # Defines the computation from inputs to outputs
      known_axes = [i for i, size in enumerate(inputs[0].shape) if size == 1]
      if len(known_axes) != 0:
          inputs[0] = tf.squeeze(inputs[0],axis=known_axes)
      outputs = tf.matmul(inputs[0], inputs[1])
      return outputs

#%%

def build_model(point_cloud, is_training, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.shape[0]
  num_point = point_cloud.shape[1]
  end_points = {}
  k = 20
  # adj_matrix = tf_util.pairwise_distance(point_cloud)
  # nn_idx = tf_util.knn(adj_matrix, k=k)
  # edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
  
  inputs = tf.keras.layers.Input((point_cloud.shape[1],3,1))
  #asd = Linear(32)(inputs)
  edge_feature = EdgeComp(k=k)(inputs)
  # with tf.variable_scope('transform_net1') as sc:
  #   transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)
  transform = input_transform_net(edge_feature, K=3)
  
  # point_cloud_transformed = tf.matmul(point_cloud, transform)
  point_cloud_transformed = MatMult()([inputs,transform])
  # adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  # nn_idx = tf_util.knn(adj_matrix, k=k)
  # edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)
  edge_feature = EdgeComp(k=k)(point_cloud_transformed)
  
  # inputs = tf.keras.layers.Input((3,3,1))
  # net = tf_util.conv2d(edge_feature, 64, [1,1],
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training,
  #                      scope='dgcnn1', bn_decay=bn_decay)
  
  x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=True, padding='valid')(edge_feature)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net1 = x

  # adj_matrix = tf_util.pairwise_distance(net)
  # nn_idx = tf_util.knn(adj_matrix, k=k)
  # edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
  
  x = EdgeComp(k=k)(x)

  # net = tf_util.conv2d(edge_feature, 64, [1,1],
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training,
  #                      scope='dgcnn2', bn_decay=bn_decay)
  # net = tf.reduce_max(net, axis=-2, keep_dims=True)
  x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net2 = x
 
  # adj_matrix = tf_util.pairwise_distance(net)
  # nn_idx = tf_util.knn(adj_matrix, k=k)
  # edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
  
  x = EdgeComp(k=k)(x)

  # net = tf_util.conv2d(edge_feature, 64, [1,1],
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training,
  #                      scope='dgcnn3', bn_decay=bn_decay)
  # net = tf.reduce_max(net, axis=-2, keep_dims=True)
  
  x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net3 = x

  # adj_matrix = tf_util.pairwise_distance(net)
  # nn_idx = tf_util.knn(adj_matrix, k=k)
  # edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
  
  x = EdgeComp(k=k)(x)
  
  # net = tf_util.conv2d(edge_feature, 128, [1,1],
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training,
  #                      scope='dgcnn4', bn_decay=bn_decay)
  # net = tf.reduce_max(net, axis=-2, keep_dims=True)
  x = tf.keras.layers.Conv2D(128, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net4 = x

  # net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1], 
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training,
  #                      scope='agg', bn_decay=bn_decay)
 
  # net = tf.reduce_max(net, axis=1, keep_dims=True)
  x = tf.keras.layers.Concatenate()([net1, net2, net3, net4])
  x = tf.keras.layers.Conv2D(1024, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=1, keepdims=True)

  # MLP on global point cloud vector
  # net = tf.reshape(net, [batch_size, -1]) 
  x = tf.keras.layers.Flatten()(x)

  # net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
  #                               scope='fc1', bn_decay=bn_decay)
  x = tf.keras.layers.Dense(512,activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
  #                        scope='dp1')
  x = tf.keras.layers.Dropout(0.5)(x)
  # net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
  #                               scope='fc2', bn_decay=bn_decay)
  x = tf.keras.layers.Dense(256,activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
  #                       scope='dp2')
  x = tf.keras.layers.Dropout(0.5)(x)
  # net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
  x = tf.keras.layers.Dense(40)(x)

  model = tf.keras.Model(inputs=inputs, outputs = x, name='dgcnn')
  model.summary()
  #return net, end_points
  return model
#%%

def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed
  print(input_feed)
  # with tf.Graph().as_default(): tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  #input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
  #pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)
  build_model(input_feed,True)
    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   feed_dict = {input_pl: input_feed, label_pl: label_feed}
    #   res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
    #   print res1.shape
    #   print res1

    #   print res2.shape
    #   print res2












