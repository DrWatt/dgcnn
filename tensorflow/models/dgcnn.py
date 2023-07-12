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
  
  inputs = tf.keras.layers.Input((point_cloud.shape[1],3,1))
  edge_feature = EdgeComp(k=k)(inputs)

  transform = input_transform_net(edge_feature, K=3)
  
  point_cloud_transformed = MatMult()([inputs,transform])

  edge_feature = EdgeComp(k=k)(point_cloud_transformed)
  
  x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=True, padding='valid')(edge_feature)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net1 = x
  
  x = EdgeComp(k=k)(x)


  x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net2 = x
  
  x = EdgeComp(k=k)(x)

  x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net3 = x
  
  x = EdgeComp(k=k)(x)
  
  x = tf.keras.layers.Conv2D(128, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=-2, keepdims=True)
  net4 = x

  x = tf.keras.layers.Concatenate()([net1, net2, net3, net4])
  x = tf.keras.layers.Conv2D(1024, kernel_size=(1,1), use_bias=True, padding='valid')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  
  x = tf.math.reduce_max(x, axis=1, keepdims=True)

  # MLP on global point cloud vector
  x = tf.keras.layers.Flatten()(x)

  x = tf.keras.layers.Dense(512,activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = tf.keras.layers.Dense(256,activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = tf.keras.layers.Dense(40)(x)

  model = tf.keras.Model(inputs=inputs, outputs = x, name='dgcnn')
  model.summary()
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

  print(input_feed)

  build_model(input_feed,True)













