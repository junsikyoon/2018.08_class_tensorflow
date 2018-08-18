# written by Sung Kyu Lim
# limsk@ece.gatech.edu

# B1: import tensorflow
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B2: variabies used
# we want uniform distribution with
# mean = 0, stddev = 1, range = [-1, 1]
# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

# B3: placeholders used
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# B4: linear model
# model = x * W + b
model = W / x + b

# B5: loss function using mean square error
cost = tf.reduce_mean(tf.square(model-y))

# B6: how we update weight and bias
# opt   = tf.train.GradientDescentOptimizer(learning_rate=0.001)
opt   = tf.train.AdamOptimizer(learning_rate=0.001)

# B7: objective during training
train = opt.minimize(cost)

# B8: session definition
train_tot = 20000
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# B9: training data
x_tr = [10, 15, 16, 1, 4, 6, 18, 12, 14, 7]
y_tr = [ 5,  2,  1, 9, 7, 8,  1,  5,  3, 6]

# B10: do training
for i in range(train_tot):
    error, _ = sess.run([cost, train], feed_dict={x: x_tr, y: y_tr})
    if i % 100 == 0:
        print(i, 'error= %.3f' % error, 'W = %.3f' % sess.run(W), 'b = %.3f' % sess.run(b))

# B11: model testing
test = 30
guess = sess.run(model, feed_dict={x: test})
print('\ntest=', test, 'guess= %.3f' % guess)