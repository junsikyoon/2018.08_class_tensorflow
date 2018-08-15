# 1: import tensorflow
import tensorflow as tf

# 2: turn off warning sign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 3: constants
T, F = 1.0, -1.0

# 4: training data
train_in  = [[T,T], [T,F], [F,T], [F,F]]
train_out = [[F], [T], [T], [F]]

# 4-1: step activation function
# step(x) = {1 if x > 0; -1 otherwise}
def step(x):
    is_greater = tf.greater(x,0)
    as_float   = tf.to_float(is_greater)
    doubled    = tf.multiply(as_float,2)
    return tf.subtract(doubled, 1)

# 5: hidden layer 1 definition
w1   = tf.Variable(tf.random_normal([2,2]))
b1   = tf.Variable(tf.zeros([2]))
out1 = tf.tanh(tf.add(tf.matmul(train_in, w1), b1))
# out1 = step(tf.add(tf.matmul(train_in, w1), b1))

# 5-1: hidden layer 2 definition
w2   = tf.Variable(tf.random_normal([2,2]))
b2   = tf.Variable(tf.zeros([2]))
out2 = tf.tanh(tf.add(tf.matmul(out1, w2), b2))

# 6: output layer definition
w3   = tf.Variable(tf.random_normal([2,1]))
b3   = tf.Variable(tf.zeros([1]))
out3 = tf.tanh(tf.add(tf.matmul(out2,w3), b3))
# out3 = step(tf.add(tf.matmul(out1,w3), b3))


# 7: error calculation
error = tf.subtract(train_out,out3)
mse   = tf.reduce_mean(tf.square(error))

# 8: training objective
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

# 9: tensorflow session preparation
sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0.01
epoch, max_epochs = 0, 1000

# 10: print w, b, and sample result
def test():
    print('\nweight 1\n', sess.run(w1))
    print('bias 1\n', sess.run(b1))
    print('\nweight 2\n', sess.run(w2))
    print('bias 2\n', sess.run(b2))
    print('\nweight 3\n', sess.run(w3))
    print('bias 3\n', sess.run(b3))
    print('output\n', sess.run(out3))
    print('mse: ', sess.run(mse), '\n')

# 11: main session
test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch: ', epoch, ', mse: ', err)
test()
