# written by Sung Kyu Lim
# limsk@ece.gatech.edu

# B1: simple add
print(2+3)

# B2: import tensorflow
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B3: tensorflow add
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

sess = tf.Session()
print(sess.run(x))

# B4: tensor flow multiply & matrix multiply
a = tf.constant([2,2])
b = tf.constant([[0,1], [2,3]])
x = tf.add(a,b)
y = tf.multiply(a,b)
z = tf.matmul([a],b)

sess = tf.Session()
x, y, z = sess.run([x, y, z])
print('x:\n', x)
print('y:\n', y)
print('z:\n', z)