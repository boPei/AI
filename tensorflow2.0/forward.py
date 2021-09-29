import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow import keras

# eliminate the warnings in tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #only provide the information when there is error

# import dataset
(x, y), _ = datasets.mnist.load_data()

# convert the array to tensor
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255. # normalize the value
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(tf.__version__)
print(type(x))
print(x.shape, y.shape, x.dtype, y.dtype)
print('the minimum value in x', tf.reduce_min(x))
print('the maximum value in x', tf.reduce_max(x))
print('the minimum value in y', tf.reduce_min(y))
print('the maximum value in y', tf.reduce_max(y))

# create a dataset based on x,y for getting the data in batches
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128) # a batch size is 128
# create a iterator to get the next batch
train_iter = iter(train_db)
sample = next(train_iter)
print('a batch sample size', sample[0].shape, sample[1].shape)

# training the network batchly process using the samples in train_db
# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# initialize the parameters
w1 = tf.Variable(tf.random.truncated_normal([784, 256],stddev=0.01)) # if the mse is None, we change a stddev =0.1
b1 = tf.Variable(np.random.randn(256), dtype=tf.float32)

w2 =tf.Variable(tf.random.truncated_normal([256, 128],stddev=0.01))
b2 =tf.Variable(np.random.randn(128), dtype=tf.float32)

w3 = tf.Variable(tf.random.truncated_normal([128, 10],stddev=0.01))
b3 = tf.Variable(np.random.randn(10), dtype=tf.float32)

lr =1e-3
for epoch in range(10):
    for step, (x, y) in enumerate(train_db): # get the samples in each batch
        #reshape the sampels
        x = tf.reshape(x, [-1, 28*28])
        # print(x)
        y_onehot = tf.one_hot(y, depth=10)

        with tf.GradientTape() as tape: # tf.GradientTape() only track the tf.Variable type
            h1 = tf.matmul(x,w1) + b1
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3

            # calculate the loss
            loss = tf.square (y_onehot - out)
            mse = tf.reduce_mean(loss)

        # calculate the grads
        grads = tape.gradient(mse, [w1, b1, w2, b2, w3, b3])
        # print(grads[0])

        # update the variables
        w1.assign_sub(lr*grads[0])
        # print(isinstance(w1, tf.Variable))

        b1.assign_sub(lr*grads[1]) #原地更新，不会改变值的类型
        w2.assign_sub(lr*grads[2])
        b2.assign_sub(lr*grads[3])
        w3.assign_sub(lr*grads[4])
        b3.assign_sub(lr*grads[5])
        # w1 = w1 - lr*grads[0]
        # b1 = b1 - lr*grads[1]
        # w2 = w2 - lr*grads[2]
        # b2 = b2 - lr*grads[3]
        # w3 = w3 - lr*grads[4]
        # b3 = b3 - lr*grads[5]

        print('step=',step)
        print('loss=',mse)






