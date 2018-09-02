# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# modified by Jun-Sik Yoon

# B1: import tensorflow and numpy
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

# B2: array for alphabet
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']

# B3: assign array index to each alphabet
# ex: 'a': 0, 'b': 1, 'c': 2, ...
num_dic = {n: i for i, n in enumerate(char_arr)}

# B4: training words and constants (5 character words)
seq_data = ['every', 'bad', 'satisfy', 'skill', 'elephant', 'university', 'computer', 'ox', 'basic', 'need', 'concatenate', 'concentrate', 'powerful', 'tensorflow', 'ibm', 'intel', 'qualcomm', 'samsung', 'nvidia', 'antidisestablishmentarianism']
n_input = n_class = 26

# B5: input encoder
def make_batch(seq_data):
    input_batch  = []
    target_batch = []
    max_len = find_max_len(seq_data)

    for seq in seq_data:
        input  = [num_dic[n] for n in seq[0:-1]]
        len = np.size(input)
        target = num_dic[seq[len]]
        array = np.concatenate((np.eye(26)[input], np.zeros((max_len-len,26), dtype=int)))
        input_batch.append(array)
#        input_batch.append(np.eye(26)[input])
        target_batch.append(target)
    return input_batch, target_batch

# find_max_len: find a maximum length among the words
def find_max_len(seq_data):
    max_len = 1
    for seq in seq_data:
        input  = [num_dic[n] for n in seq[0:-1]]
        len = np.size(input)
        is_greater = np.greater(len,max_len)
        if is_greater:
            max_len = len
    return max_len

max_length = find_max_len(seq_data)

# B6: global parameters
learning_rate = 0.01
n_hidden      = 128
total_epoch   = 100

# B7: placeholders and variables
# note that Y, output label, is 1-dimensional
X = tf.placeholder(tf.float32, [None, max_length, n_input])
Y = tf.placeholder(tf.int32, [None])
w = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# B8: twoo RNN cells and their deep RNN network
# we use dropout in cell 1
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)

# cell 2
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(cell2, X, dtype=tf.float32)

# B9: output re-ordering and trimming necessary for RNN
# [batch_size, n_stage, n_hidden] ->
# [n_stage, batch_size, n_hidden] ->
# [batch_size, n_hidden]
# model output becomes one-hot encoding with 26 entries
outputs = tf.transpose(outputs, [1,0,2])
outputs = outputs[-1]
model = tf.matmul(outputs, w) + b

# B10: loss function and optimizer
# note that we use sparse version of SCEL for loss
# to better handle 1-dimensional label
# note the difference in size between logits and labels
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# B11: training session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, error = sess.run([optimizer, loss], feed_dict={X: input_batch, Y: target_batch})
    print('epoch: %04d' % epoch, 'error = %.4f' % error)
print()

# B12: testing the final RNN model
# note that our model output is floating point tensor
# and label is integer, not one-hot encoding
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
input_batch, target_batch = make_batch(seq_data)
guess = sess.run(prediction, feed_dict={X: input_batch})

for i, seq in enumerate(seq_data):
    print(seq[0:-1], char_arr[guess[i]])