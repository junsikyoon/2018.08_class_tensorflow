# written by Sung Kyu Lim
# limsk@ece.gatech.edu

# B1: import tensorflow
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B2: constants
T, F = 1., -1.
bias = 1.

# B3: training data (2-input OR function)
train_in  = [[T,T,bias], [T,F,bias], [F,T,bias], [F,F,bias]]
train_out = [[T], [T], [T], [F]]

# B4: weight matrix definition
w = tf.Variable(tf.random_normal([3,1]))

# B5: step activation function
# step(x) = {1 if x > 0; -1 otherwise}
def step(x):
    is_greater = tf.greater(x,0)
    as_float   = tf.cast(is_greater, tf.float32)
    doubled    = tf.multiply(as_float, 2)
    return tf.subtract(doubled,1)

# B6: output and error definition
output = step(tf.matmul(train_in, w))
error  = tf.subtract(train_out, output)
mse    = tf.reduce_mean(tf.square(error))

# B7: weight update definition
# train_in: 4x3 tensor
# error: 4x1 tensor
# we tranpose train_in to obtain delta (3x1 tensor)
# weight change is done with tf.assign
delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w,delta))

# B8: tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0
epoch, max_epochs = 0, 100

# B9: print w, b, and sample result
def test():
    print('\nweight/bias\n', sess.run(w))
    print('output\n', sess.run(output))
    print('mse: ', sess.run(mse), '\n')

# B10: main session
test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse,train])
    print('epoch:', epoch, 'mse:', err)
test()