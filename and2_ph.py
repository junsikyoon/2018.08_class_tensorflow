# tensorflow_class
# assignments, lecture notes, examples

# 1: import tensorflow
import tensorflow as tf

# 2: turn off warning sign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 3: constants
T, F = 1.0, -1.0
bias = 1.0

# 4: training data (placeholder)
train_in  = tf.placeholder(tf.float32, [None,3])
train_out = tf.placeholder(tf.float32, [None,1])

# 5: weight
w = tf.Variable(tf.random_normal([3,1]))

# 6: step activation function
# step(x) = {1 if x > 0; -1 otherwise}
def step(x):
    is_greater = tf.greater(x,0)
    as_float   = tf.to_float(is_greater)
    doubled    = tf.multiply(as_float,2)
    return tf.subtract(doubled, 1)

# 7: output and error definitions
output = step(tf.matmul(train_in,w))
# output = tf.tanh(tf.matmul(train_in,w))
error  = tf.subtract(train_out, output)
mse = tf.reduce_mean(tf.square(error))

# train = tf.train.GradientDescentOptimizer(0.02).minimize(mse)

# 8: weight update definition
# train_in: 4x3 tensor
# error   : 4x1 tensor
# we have to transpose train_in (4x3 -> 3x4)
# weight change is done with tf.assign
delta = tf.matmul(train_in, error, transpose_a = True)
train = tf.assign(w, tf.add(w,delta))

# 9: tensorflow session
sess = tf.Session()

in_tr  = [[T,T,bias], [T,F,bias], [F,T,bias], [F,F,bias]]
out_tr = [[T], [F], [F], [F]]

sess.run(tf.global_variables_initializer())
err, target = 1, 0.01
epoch, max_epochs = 0, 100

# 10: print w, b, and sample result
def test():
    print('\nweight/bias\n', sess.run(w))
    print('output\n', sess.run(output))
    print('mse: ', sess.run(mse), '\n')

# 11: main session
# test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, result, _ = sess.run([mse, output, train], feed_dict={train_in:in_tr, train_out:out_tr})
    print('epoch:', epoch, 'mse', err)
print('output:\n', result, '\n')
# test()
