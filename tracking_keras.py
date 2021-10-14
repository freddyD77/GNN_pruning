from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
import numpy as np
import sys
import glob
from datetime import datetime
import random
from copy import deepcopy

from qkeras import print_qstats
from qkeras import QActivation
from qkeras import QDense
from qkeras import quantized_bits

from keras.layers import Dense
from keras.activations import tanh

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import prune

import os
import zipfile
from tensorflow.keras.models import load_model
import tempfile

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

class InputNetwork(keras.layers.Layer):
    def __init__(self, hidden_dim=8, **pruning_params):
        super(InputNetwork, self).__init__()
        '''self.linear_1 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.act1 = QActivation("quantized_tanh(14,4,1)")'''
        self.linear_1 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.act1 = tanh

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.act1(x)

class EdgeNetwork(keras.layers.Layer):
    def __init__(self, hidden_dim=8, **pruning_params):
        super(EdgeNetwork, self).__init__()
        '''self.linear_1 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.linear_2 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.linear_3 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.linear_4 = QDense(1, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.act1 = QActivation("quantized_tanh(14,4,1)")
        self.act2 = QActivation("quantized_tanh(14,4,1)")
        self.act3 = QActivation("quantized_tanh(14,4,1)")'''
        self.linear_1 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.linear_2 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.linear_3 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.linear_4 = prune.prune_low_magnitude(Dense(1), **pruning_params)
        self.act1 = tanh
        self.act2 = tanh
        self.act3 = tanh

    def call(self, inputs, edge_index):
        x = inputs
        x1 = tf.gather_nd(x,tf.transpose([edge_index[0]]))
        x2 = tf.gather_nd(x,tf.transpose([edge_index[1]]))
        edge_inputs = tf.concat([x1, x2],1)
        x = self.linear_1(edge_inputs)
        x = self.act1(x)
        x = self.linear_2(x)
        x = self.act2(x)
        x = self.linear_3(x)
        x = self.act3(x)
        x = self.linear_4(x)
        return x

class NodeNetwork(keras.layers.Layer):
    def __init__(self, hidden_dim=8, **pruning_params):
        super(NodeNetwork, self).__init__()
        '''self.linear_1 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.linear_2 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.linear_3 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.linear_4 = QDense(hidden_dim, kernel_quantizer=quantized_bits(14,4,1), bias_quantizer=quantized_bits(14,4,1))
        self.act1 = QActivation("quantized_tanh(14,4,1)")
        self.act2 = QActivation("quantized_tanh(14,4,1)")
        self.act3 = QActivation("quantized_tanh(14,4,1)")
        self.act4 = QActivation("quantized_tanh(14,4,1)")'''
        self.linear_1 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.linear_2 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.linear_3 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.linear_4 = prune.prune_low_magnitude(Dense(hidden_dim), **pruning_params)
        self.act1 = tanh
        self.act2 = tanh
        self.act3 = tanh
        self.act4 = tanh

    def call(self, inputs, e, edge_index):
        x = inputs
        x1 = tf.gather_nd(x,tf.transpose([edge_index[0]]))
        x2 = tf.gather_nd(x,tf.transpose([edge_index[1]]))
        ex1 = e * x1
        ex2 = e * x2
        if x.shape[0] == None:
            xshape0 = 100
        else:
            xshape0 = x.shape[0]
        mi = tf.scatter_nd(tf.transpose([edge_index[1]]), ex1, shape=(xshape0,x.shape[1]))
        mo = tf.scatter_nd(tf.transpose([edge_index[0]]), ex2, shape=(xshape0,x.shape[1]))
        node_inputs = tf.concat([mi, mo, x],1)
        x = self.linear_1(node_inputs)
        x = self.act1(x)
        x = self.linear_2(x)
        x = self.act2(x)
        x = self.linear_3(x)
        x = self.act3(x)
        x = self.linear_4(x)
        return self.act4(x)

class agnn(keras.layers.Layer):#keras.Model):
    def __init__(self, hidden_dim=8,n_graph_iters=4):#, **pruning_params):
        super(agnn, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.hidden_dim = hidden_dim
        pruning_params = {
            "pruning_schedule":
                pruning_schedule.ConstantSparsity(0.5, begin_step=0, frequency=10)
            }
        self.inputNet = InputNetwork(self.hidden_dim, **pruning_params)
        self.edgeNet = EdgeNetwork(self.hidden_dim, **pruning_params)
        self.nodeNet = NodeNetwork(self.hidden_dim, **pruning_params)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_dim': self.hidden_dim,
            'n_graph_iters': self.n_graph_iters,
        })
        return config

    def call(self, inputs0):
        """Apply forward pass of the model"""
        inputs = inputs0[0][0]
        edge_index = tf.cast(inputs0[1][0], tf.int32)

        # Apply input network to get hidden representation
        x = self.inputNet(inputs)

        for i in range(self.n_graph_iters):

            # Previous hidden state
            x0 = x

            # Apply edge network
            e = tf.nn.sigmoid(self.edgeNet(x, edge_index))
            #print(e.shape)

            # Apply node network
            x = self.nodeNet(x, e, edge_index)

            # Residual connection
            x = x + x0

        # Apply final edge network
        e = self.edgeNet(x, edge_index)
        return e

def makeBatches(dataset, batches):
    newdataset=[]
    previous_nodes=0
    for i, (x, y) in enumerate(dataset):
        for edge in x[1]:
            for index in range(len(edge)):
                edge[index]+=previous_nodes
        if i % batches == 0:
            newx = [x[0],x[1]]
            newy = y
        else:
            newx[0] = np.concatenate((newx[0], x[0]), axis=0)
            newx[1] = np.concatenate((newx[1], x[1]), axis=1)
            newy = np.concatenate((newy, y), axis=0)
        previous_nodes += len(x[0])
        if (i+1) % batches == 0:
            newdataset.append([newx,newy])
            previous_nodes = 0
    return newdataset

def makeBatches2(dataset, batches):
    newdataset=[]
    previous_nodes=0
    for i, (x, y) in enumerate(dataset):
        for edge in x[1][0]:
            for index in range(len(edge)):
                edge[index]+=previous_nodes
        if i % batches == 0:
            newx = [x[0],x[1]]
            newy = y
        else:
            concat1 = np.concatenate((newx[0][0], x[0][0]), axis=0)
            newx[0]=concat1[np.newaxis, :, :]
            concat2 = np.concatenate((newx[1][0], x[1][0]), axis=1)
            newx[1]=concat2[np.newaxis, :, :]
            newy = np.concatenate((newy, y), axis=0)
        previous_nodes += len(x[0][0])
        if (i+1) % batches == 0:
            newdataset.append([newx,newy])
            previous_nodes = 0
    return newdataset

def get_gzipped_model_size2(model):
  # Returns size of gzipped model, in bytes.

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)

# load the dataset
mypath = '../data6/parsed_In_Iteration6'
mypath2 = '../data6/parsed_D0_Iteration6'
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]

files = 100#2500
dataset = []
valid_dataset=[]
validf = 1.1

for i in range(int(files*validf)):
    event = onlyfiles[i]
    data = np.load('../data6/parsed_In_Iteration6/'+event)
    inputs0 = data['X']
    edges = data['edge_index']
    inputs0 = inputs0[np.newaxis, :, :]#
    edges = edges[np.newaxis, :, :]#
    pid = data['pid']
    start, end = edges[0]#
    y = np.logical_and(pid[start] != -1, pid[start] == pid[end])
    if i >= files:
        valid_dataset.append([[inputs0, edges],y])
    else:
        dataset.append([[inputs0, edges],y])

for i in range(int(files*validf)):
    event = onlyfiles2[i]
    data = np.load('../data6/parsed_D0_Iteration6/'+event)
    inputs0 = data['X']
    edges = data['edge_index']
    inputs0 = inputs0[np.newaxis, :, :]#
    edges = edges[np.newaxis, :, :]#
    pid = data['pid']
    start, end = edges[0]#
    y = np.logical_and(pid[start] != -1, pid[start] == pid[end])
    if i >= files:
        valid_dataset.append([[inputs0, edges],y])
    else:
        dataset.append([[inputs0, edges],y])

batches = 10
batch_valid_dataset = makeBatches2(deepcopy(valid_dataset),batches)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_metric = tf.keras.metrics.Accuracy()

# define the keras model
dim = 8#8
iters = 4#4
epochs = 10
#spec = tf.RaggedTensorSpec(shape=(None, None),dtype=tf.float32, ragged_rank=1)
nodesIn = keras.Input(shape=(None,4), batch_size=1)
edgesIn = keras.Input(shape=(2,None), batch_size=1)
inputPlaceHolder=[nodesIn,edgesIn]
outputPlaceHolder = agnn(dim, iters)(inputPlaceHolder)
print(outputPlaceHolder.shape)
model = keras.Model(inputPlaceHolder, outputPlaceHolder)
#model = agnn(dim, iters)#, **pruning_params)
model(dataset[0][0])
model.summary()
#print(model.trainable_weights)
model.optimizer = optimizer
step_callback = tfmot.sparsity.keras.UpdatePruningStep()
step_callback.set_model(model)
log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir="pruneLog2") # Log sparsity and other metrics in Tensorboard.
log_callback.set_model(model)

# Iterate over epochs
step_callback.on_train_begin()
for epoch in range(epochs):
    # Iterate over the batches of a dataset.
    log_callback.on_epoch_begin(epoch=-1)
    epoch_loss = 0
    epoch_acc = 0
    total_edges = 0
    correct_edges = 0
    random.shuffle(dataset)
    batch_dataset = makeBatches2(deepcopy(dataset),batches)
    for step, (x, y) in enumerate(batch_dataset):
        step_callback.on_train_batch_begin(batch=-1)
        # Open a GradientTape.
        with tf.GradientTape() as tape:
            # Forward pass.
            logits = model(x, training=True)
            # Loss value for this batch.
            loss_value = loss_fn(tf.reshape(y,[1,-1]), tf.reshape(logits,[1,-1]))
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        # Update the weights of the model.
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        logits = (logits > 0.5)
        epoch_loss += loss_value
        total_edges += len(y)
        for i in range(len(y)):
            if y[i] == logits[i][0]:
                correct_edges += 1
    step_callback.on_epoch_end(batch=-1)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + "Epoch %d" % epoch)
    print("Loss = %.4f" % (epoch_loss/len(batch_dataset)))
    print("Train Acc = %.4f" % (correct_edges/(total_edges)))

    total_edges = 0
    correct_edges = 0
    for step, (x, y) in enumerate(batch_valid_dataset):
        logits = model(x)
        logits = (logits > 0.5)
        total_edges += len(y)
        for i in range(len(y)):
            if y[i] == logits[i][0]:
                correct_edges += 1
    print("Test Acc = %.4f" % (correct_edges/(total_edges)))

model.save_weights("model.h5")#trackingGNN_16_4_100e_10b_5000f.h5")

print("Size of gzipped pruned model without stripping: %.2f bytes" % (get_gzipped_model_size2(model)))
model_for_export = tfmot.sparsity.keras.strip_pruning(model)
print("Size of gzipped pruned model with stripping: %.2f bytes" % (get_gzipped_model_size2(model_for_export)))



