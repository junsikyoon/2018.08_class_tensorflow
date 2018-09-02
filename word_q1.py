# written by Sung Kyu Lim
# limsk@ece.gatech.edu

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

# B4: training words and constants
seq_data = ['body', 'dial', 'open', 'rank', 'need', 'wise', 'item', 'jury', 'path', 'ease']
n_input = n_class = 26
n_stage = 3

test_data = ['bod', 'bad', 'boe', 'nar', 'nef', 'zzz']

# B5: input encoder
def make_batch(seq_data):
    input_batch  = []
    target_batch = []

    for seq in seq_data:
        input  = [num_dic[n] for n in seq[0:-1]]
        target = num_dic[seq[3]]
        input_batch.append(np.eye(26)[input])
        target_batch.append(target)
    return input_batch, target_batch

def test(test_data):
    input_batch = []

    for seq in test_data:
        input = [num_dic[n] for n in seq]
        input_batch.append(np.eye(26)[input])
    return input_batch

# B6: global parameters
learning_rate = 0.01
n_hidden      = 128
total_epoch   = 100

# B7: placeholders and variables
# note that Y, output label, is 1-dimensional
X = tf.placeholder(tf.float32, [None, n_stage, n_input])
Y = tf.placeholder(tf.int32, [None])
w = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# B8: twoo RNN cells and their deep RNN network
# we use dropout in cell 1
# cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)

cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

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
    print(seq[0:3], char_arr[guess[i]])

# B13: use test data
input_batch = test(test_data)
guess = sess.run(prediction, feed_dict={X: input_batch})

print('\n')
for i, seq in enumerate(test_data):
    print(seq[0:3], char_arr[guess[i]])
