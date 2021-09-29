import tensorflow as tf
import numpy as np

outputs = tf.random.uniform([4, 10])
y = tf.constant([2, 3, 2, 0])
y = tf.one_hot(y, depth=10)
loss = tf.keras.losses.mse(y, outputs)
loss = tf.reduce_mean(loss)

x = tf.random.normal([4, 4, 4, 3])

x = tf.random.uniform([28, 28], maxval=10, dtype=tf.float32)

x = tf.random.uniform([1, 28, 28, 1], maxval=10, dtype=tf.int32)

# 4.7.4 Data multiplication
b = tf.constant([1, 2])
b_dims = tf.expand_dims(b, axis=0)
b_dims_expand=tf.tile(b_dims, multiples=[2, 1])

x = tf.range(4)
x_reshape=tf.reshape(x, [2, 2])

# 4.9 Mathematics
a = tf.range(5)
b = tf.constant(2)
print('a divided by b is', a/b)

# a mod b is
print('a mod b is', a % b)


# 4.9.2 Square
x = tf.range(4)
print('the square of each number in x is',tf.square(x))
print('the specific power of each number in x is', tf.pow(x, 3))

x = tf.constant([1, 3, 4, 6],dtype=tf.float32)
print("the root of each number in x is", x**(0.5))

# 4.9.3 Exponent and log
x = tf.constant([1., 2., 3.])
ex = tf.exp(x)
print('exponent of x is', ex)
print('the log of x is', tf.math.log(ex)) # the result is natural log which is the result of log_e(x)

# how to compute the log_a(x) for example, log_3(x)

b = tf.math.log(x)/tf.math.log(3.)
print('the result of log_3(x) is ', b)