import os
import os.path as path

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

print(tf.__version__)

MODULE_NAME = 'mnist_convnet'
EPOCHS = 2
BATCH_SIZE = 128

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                     activation='relu', input_shape=[28, 28, 1]))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

def train_model(model,x_train,y_train,x_test,y_test):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test))



def export_model(saver, model, input_node_names, output_node_names):
    tf.train.write_graph(K.get_session().graph_def, 'knowlegeTracing', MODULE_NAME + "_graph.pbtxt")
    saver.save(K.get_session(), 'knowlegeTracing/' + MODULE_NAME + '.chkp')

    freeze_graph.freeze_graph('knowlegeTracing/' +MODULE_NAME+'_graph.pbtxt',
                              None, False, 'knowlegeTracing/'+MODULE_NAME+'.chkp',
                              output_node_names,'knowlegeTracing/frozen_'+MODULE_NAME+'.pb', True, '')


    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('knowlegeTracing/frozen_'+MODULE_NAME+'.pb','rb') as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,input_node_names,[output_node_names],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('knowlegeTracing/opt_'+MODULE_NAME+'.pb','wb') as f:
        f.write(output_graph_def.SerializeToString())
    print('graph saved')


# def main():
#     if not os.path.exists('knowlegeTracing'):
#         os.makedirs('knowlegeTracing')
#     x_train, y_train, x_test, y_test = load_data()
#     model = build_model()
#     train_model(model, x_train, y_train, x_test, y_test)
#     export_model(tf.train.saver(), model, ['conv2d_1_input'], 'dense_2/softmax')
#
# if __name__ == '__main__':
#     main()

if not os.path.exists('out'):
    os.makedirs('out')
x_train, y_train, x_test, y_test = load_data()
model = build_model()
# train_model(model, x_train, y_train, x_test, y_test)
export_model(tf.train.Saver(), model, ['conv2d_1_input'], 'dense_2/softmax')

