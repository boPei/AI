import tensorflow as tf
import numpy as np

print(tf.__version__)

# 4.4 Create Tensor
# 4.4.1 Create the tensors from list and Array
list_tensor = tf.convert_to_tensor([1 , 2.])

array_tensor = tf.convert_to_tensor(np.array([[2,3],[1.,2.]]))

# 4.4.2 Create all 0s and 1s vector
allZeros_tensor = tf.zeros([2])
allOnes_tensor = tf.ones([2, 2])

zerosLike = tf.zeros_like(allOnes_tensor)
onesLike = tf.ones_like(allZeros_tensor)

# 4.4.3 Create the self-defined tensors
selfDefined_tensor = tf.fill([2, 1], -1)  # create a tensor with a shape of 2,1
selfDefined_tensor_1=tf.fill([], -1) #create a tensor which stores a scalar

# 4.4.4 Create a known-distribution tensor
normalDist = tf.random.normal([2, 2], mean=0, stddev=1) # create a 2x2 matrix with a mean of 0 and standard deviation of 1
uniformDist = tf.random.uniform([2, 2], minval=0, maxval=1, dtype=tf.float32) # Create a 2x2 matrix uniformly distributed matrix

# 4.4.5 Create Sequences
rangeVal = tf.range(10, delta=1) #Create a integer sequence from 0 to 10-1 with a step of delta
rangeVal_1 = tf.range(10, delta=3)



