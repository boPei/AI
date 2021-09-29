import tensorflow as tf
import numpy as np
import os
import sys
print(tf.__version__)
print(sys.version_info)
print(sys.version_info[0])

dataDir=os.path.join(os.path.abspath(os.path.dirname(__file__)),'data')
print(dataDir)

'''
split knowlegeTracing each sentence into words list and replace the '\n' with '<eos>'

input argument: filename
'''
def _read_words(filename):
    with tf.io.gfile.GFile(filename,'r') as f:
        if sys.version_info[0]>=3:
            return f.read().replace('/n','<eos>').split()
        else:
            return f.read().decode('utf-8').replace('/n','<eos>').split()


data=_read_words(os.path.join(dataDir,'ptb.test.txt'))
print(data)


